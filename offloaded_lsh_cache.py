from transformers import StaticCache, OffloadedStaticCache, PretrainedConfig
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union



class OffloadedLSHStaticCache(StaticCache):
    """
    Static cache class to be used with `torch.compile(model)` that offloads to the CPU or
    another device.

    Args:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize
            the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`Union[str, torch.device]`):
            The device on which the cache should be initialized. Should be the same as the
            layer device.
        dtype (`torch.dtype`, *optional*):
            The default `dtype` to use when initializing the cache.
        offload_device (`Union[str, torch.device]`, *optional*, defaults to `cpu`):
            The device to offload to. Defaults to CPU.
        layer_device_map (`Dict[int, Union[str, torch.device, int]]`, *optional*):
            Mapping between the layers and its device. This is required when you are manually initializing the cache and the model is splitted between differents gpus.
            You can know which layers mapped to which device by checking the associated device_map: `model.hf_device_map`.

    Attributes:
        key_cache (`List[torch.Tensor]`):
            Off-loaded key cache tensors. First one will be on device, where-as the others are
            off-loaded.
        value_cache (`List[torch.Tensor]`):
            Off-loaded value cache tensors. First one will be on device, where-as the others are
            off-loaded.
        max_batch_size (`int`):
            The maximum batch size with which this cache can be used.
        max_cache_len (`int`):
            The maximum sequence length with which this cache can be used.
        device (`torch.device`):
            The device on which the cache is used.
        offload_device (`torch.device`):
            The device used to offload to.
        dtype (`torch.dtype`):
            The `dtype` used to initializing the cache.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, OffloadedStaticCache

        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

        >>> inputs = tokenizer(text="My name is GPT2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = OffloadedStaticCache(config=model.config, max_batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> past_kv_length = outputs.past_key_values # access cache filled with key/values from generation
        ```
    """

    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        max_cache_len: Optional[int],
        device: Union[str, torch.device],
        dtype: Optional[torch.dtype] = None,
        offload_device: Union[str, torch.device] = torch.device("cpu"),
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        self.device = torch.device(device) if layer_device_map is None else layer_device_map[0]
        self.offload_device = torch.device(offload_device)
        self.dtype = dtype if dtype is not None else torch.float32

        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        head_dim = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads

        num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )

        cache_shape = (max_batch_size, num_key_value_heads, self.max_cache_len, head_dim)

        # Create offloaded CPU tensors.
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        for i in range(config.num_hidden_layers):
            # First layer is always on-device.
            device = self.device if i == 0 else self.offload_device

            key_cache, value_cache = self._create_key_value_cache_tensors(cache_shape, device)

            self.key_cache.append(key_cache)
            self.value_cache.append(value_cache)

        # Create device tensors.
        self._device_key_cache: List[torch.Tensor] = []
        self._device_value_cache: List[torch.Tensor] = []

        for i in range(2): # MLIU: why 2? to alternate between two on-device caches?
            key_cache, value_cache = self._create_key_value_cache_tensors(cache_shape, self.device)

            self._device_key_cache.append(key_cache)
            self._device_value_cache.append(value_cache)

        # TODO(mliu) add LSH hash table for keys
        lsh_dim = 8
        proj_mat_shape = (head_dim, lsh_dim)
        self.proj_mat = torch.randn(proj_mat_shape, device=self.device, dtype=self.dtype)
        self._device_key_hash: List[torch.Tensor] = []

        # For backwards compatibility.
        # TODO(gante): Remove this.
        self._seen_tokens = 0

        # Create new CUDA stream for parallel prefetching.
        self._prefetch_stream = torch.cuda.Stream() if self.device.type == "cuda" else None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, *optional*):
                Additional arguments for the cache subclass. The `OffloadedStaticCache` needs the
                `cache_position` input to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """

        if layer_idx == 0:
            # Update seen tokens.
            # TODO(gante): Remove this.
            self._seen_tokens += key_states.shape[-2] 
            # MLIU: shape of key_states is (batch_size, num_heads, seq_len, head_dim) ?
            # MLIU: key_states.shape[-2] is the sequence length

            # Always there.
            k_out = self.key_cache[0]
            v_out = self.value_cache[0]
        else:
            # Wait for prefetch stream.
            if self._prefetch_stream is not None:
                torch.cuda.default_stream(self.device).wait_stream(self._prefetch_stream)

            k_out = self._device_key_cache[layer_idx & 1] # MLIU: &1 is to alternate between two on-device caches?
            v_out = self._device_value_cache[layer_idx & 1]

        # TODO(mliu) modify the 
        self._prefetch_layer(layer_idx + 1)

        # TODO(mliu) get LSH related cache kw args here
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None

        if cache_position is None:
            k_out.copy_(key_states)
            v_out.copy_(value_states)

            # Copy the values to the offloaded device as well.
            if layer_idx == 0:
                self.key_cache[layer_idx].copy_(key_states.to(self.offload_device))
                self.value_cache[layer_idx].copy_(value_states.to(self.offload_device))
        else:
            # Note: here we use `tensor.index_copy_(dim, index, tensor)` that is equivalent to
            # `tensor[:, :, index] = tensor`, but the first one is compile-friendly and it does
            # explicitly an in-place operation, that avoids copies and uses less memory.
            try:
                k_out.index_copy_(2, cache_position, key_states)
                v_out.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                # The operator 'aten::index_copy.out' is not currently implemented for the MPS
                # device.
                k_out[:, :, cache_position] = key_states
                v_out[:, :, cache_position] = value_states

            # Copy the values to the offloaded device as well.
            if layer_idx != 0:
                cache_position = cache_position.to(self.offload_device)
                key_states = key_states.to(self.offload_device)
                value_states = value_states.to(self.offload_device)

                try:
                    self.key_cache[layer_idx].index_copy_(2, cache_position, key_states)
                    self.value_cache[layer_idx].index_copy_(2, cache_position, value_states)
                except NotImplementedError:
                    # The operator 'aten::index_copy.out' is not currently implemented for the MPS
                    # device.
                    self.key_cache[layer_idx][:, :, cache_position] = key_states
                    self.value_cache[layer_idx][:, :, cache_position] = value_states

        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""

        # TODO(gante): Remove this.
        return self._seen_tokens

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""

        return self.max_cache_len

    def reset(self) -> None:
        """Resets the cache values while preserving the objects."""

        # For backwards compatibility.
        # TODO(gante): Remove this.
        self._seen_tokens = 0

        # Zero out cache.
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address.
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()

    @property
    def seen_tokens(self) -> int:
        # For backwards compatibility.
        # TODO(gante): Remove this.
        return self._seen_tokens

    def _create_key_value_cache_tensors(
        self, shape: Tuple[int, ...], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Creates K/V cache tensors on a device. Pins memory for CPU tensors. Marks them as static
        addresses for non-CPU tensors.

        Args:
            shape (`Tuple[int, ...]`): Shape.
            device (`torch.device`): Device.

        Returns:
            Key and value cache tensors as a tuple.
        """

        is_cpu_device = device == torch.device("cpu")

        key_cache = torch.zeros(shape, dtype=self.dtype, device=device, pin_memory=is_cpu_device)
        value_cache = torch.zeros(shape, dtype=self.dtype, device=device, pin_memory=is_cpu_device)

        # Note: `mark_static_address` is used to tag the cache as a fixed data pointer,
        # preventing compiled graph breaks when updating the cache.
        torch._dynamo.mark_static_address(key_cache)
        torch._dynamo.mark_static_address(value_cache)

        return key_cache, value_cache

    def _prefetch_layer(self, layer_idx: int) -> None:
        """Prefetch a layer to the device. Needs to be called in order of layer indices."""

        # Don't fetch layers that do not exist.
        if layer_idx >= len(self.key_cache):
            return

        # Alternate between two on-device caches.
        if self._prefetch_stream is not None:
            with torch.cuda.stream(self._prefetch_stream):
                self._prefetch_layer_in_context(layer_idx)
        else:
            self._prefetch_layer_in_context(layer_idx)


    # TODO: pass in the current key hash as argument
    def _prefetch_layer_in_context(self, layer_idx: int) -> None:
        """Performs the actual copy of the layer to device cache."""

        # MLIU: old code
        # self._device_key_cache[layer_idx & 1].copy_(self.key_cache[layer_idx], non_blocking=True)
        # self._device_value_cache[layer_idx & 1].copy_(self.value_cache[layer_idx], non_blocking=True)

        # TODO(mliu) prefetch selectively based on LSH hash hamming distance
        hamming_distances = self._hamming_dist(self._hash_fn(self.key_cache[layer_idx]), current_key_hash)
        # sort the distances based on the hash table
        sorted_indices = torch.argsort(hamming_distances)
        # take the top max_cache_shape indices
        top_indices = sorted_indices[:self.get_max_cache_shape()]
        # copy the top indices to the device cache
        self._device_key_cache[layer_idx & 1].index_copy_(2, top_indices, self.key_cache[layer_idx], non_blocking=True)
        self._device_value_cache[layer_idx & 1].index_copy_(2, top_indices, self.value_cache[layer_idx], non_blocking=True)


    def _hash_fn(self, key_states: torch.Tensor) -> torch.Tensor:
        """Hash function to be used for LSH cache."""
        proj = torch.matmul(key_states, self.proj_mat)
        # use sign() and convert to binary (dtype=int)
        return ((proj.sign() + 1) // 2).to(torch.int)
    
    def _hamming_dist(self, hash1: torch.Tensor, hash2: torch.Tensor) -> torch.Tensor:
        """Calculate the hamming distance between two hash tensors."""
        return torch.sum(hash1 != hash2, dim=-1)
    