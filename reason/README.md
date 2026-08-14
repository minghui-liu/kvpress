# Evaluation

This directory contains a set of scripts to evaluate the performance of different presses on different reasoning datasets. We currently support the following datasets:
- GSM8K ([openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k))
- StrategyQA ([ChilleD/StrategyQA](https://huggingface.co/datasets/ChilleD/StrategyQA))
- FOLIO ([yale-nlp/folio](https://huggingface.co/datasets/yale-nlp/folio))
- LogiQA ([lucasmccabe/logiqa](https://huggingface.co/datasets/lucasmccabe/logiqa))
- OpenbookQA ([allenai/openbookqa](https://huggingface.co/datasets/allenai/openbookqa))
- MATH-500 ([HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500))
- AIME25 ([math-ai/aime25](https://huggingface.co/datasets/math-ai/aime25))
- GPQA ([Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa))
- GPQA Diamond ([Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa), config: `gpqa_diamond`)


Please refer to the huggingface page and the original paper of each dataset for more information on how the Hugging Face dataset was generated.

## Answer grading

Answer grading is benchmark-aware; a single exact-string comparison or regex is not sufficient across these tasks.

- GSM8K uses exact rational arithmetic after final-answer extraction. Equivalent forms such as `1/2`, `1 / 2`, `0.5`, and `0.500000` match, as do harmless currency, thousands-separator, scientific-notation, and unit variations. This extends the [official GSM8K final-answer convention](https://github.com/openai/grade-school-math), where the gold answer follows `####`.
- AIME24 and AIME25 use the same exact numeric representation but require an integer result. Leading zeroes and integral decimal/fraction forms are equivalent. AIME24's dataset stores a full solution, so its formatter first extracts the boxed gold answer.
- MATH-500 uses [Math-Verify](https://github.com/huggingface/Math-Verify): answer extraction, LaTeX normalization/parsing, conversion to SymPy, and symbolic/numeric equivalence. The original MATH evaluator only applied [string normalization](https://github.com/hendrycks/math/blob/main/modeling/math_equivalence.py); Math-Verify covers substantially more sets, intervals, equations, tuples, matrices, and equivalent expressions.
- DROP reports exact match and token F1 using the [official DROP evaluator](https://github.com/allenai/allennlp-models/blob/main/allennlp_models/rc/tools/drop.py), with exact rational normalization added for numeric forms. The `ucinlp/drop` dataset flattens the original and validated annotations into one `answers_spans.spans` list, so the scorer takes the maximum metric over those alternative annotations rather than requiring every entry simultaneously.
- CommonsenseQA, OpenBookQA, ReClor, LogiQA, GPQA, and GPQA-Diamond use strict choice accuracy while accepting an equivalent letter or one-based option number. GPQA raw zero-based gold indices are normalized before comparison.
- StrategyQA and FOLIO use strict categorical accuracy with casing/format normalization. `yes`/`no` map to booleans, and `unknown`/`undetermined` map to FOLIO's `Uncertain` class.

All scorers reject prediction/gold length mismatches instead of silently truncating with `zip`. Math scorers receive the full response so the parser can prioritize the last explicit or boxed answer. The implementation never executes model output with Python `eval`; unsupported or ambiguous expressions are marked incorrect rather than run as code.

## Installation
Follow the instuction in the README file of the root folder of KVPress to install KVPress.

If you want to install flash-attn but the process is too slow, then this post might be helpful:
https://github.com/simonw/til/blob/main/python/installing-flash-attention.md


## Usage

To evaluate a press on a dataset, you can run the following command:
```
cd reason
```

```bash
python evaluate.py --dataset <dataset_name> --data_dir <data_dir> --data_split <data_split> --model <model_name> --press_name <press_name> --cache_budget <budget> --max_new_tokens <max_new_tokens>
```

For instance,
```bash
python evaluate.py --dataset gsm8k --data_split test --model meta-llama/Meta-Llama-3.1-8B-Instruct --press_name knorm --cache_budget 128  --max_new_tokens 512
```

To run a small subset of the dataset you can set `--fraction <float between 0 and 1.0>`. 


- Results (predictions & metrics) are saved in the `results` directory. 
- All available presses are listed in the `PRESS_DICT` variable in `evaluate.py`. 
- Additional arguments are --device, --fraction, --max_new_tokens and --max_context_length. For more information, run `python evaluate.py --help`
- Finally we also provide a bash script `evaluate.sh` to facilitate the evaluation of multiple presses (1 per GPU) with different compression ratios.


## How to add a dataset

Each dataset script should have a script that is structured as follows:

```bash
$dataset.py
├── $dataset_formatter
├── $dataset_scorer
├── $dataset_extractor
├── metric functions
```

Where:
- $dataset_formatter is the formatter function that returns a input text string and a answer string
- $dataset_scorer is the function that returns a dictionary of scores of different metrics
- $dataset_extractor is the function that extracts the model's answer from the model's response string. `utils.py` has a `default_extractor` function that you can use if the dataset does not need any special extraction method.
- metric functions are the functions for different metrics used to evaluate the answers

Finally add the new dataset's formatter, extractor and scorer to the corresponding dictionaries in `evaluate.py`.
