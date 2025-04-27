# Evaluation

This directory contains a set of scripts to evaluate the performance of different presses on different reasoning datasets. We currently support the following datasets:
- GSM8K ([openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k))
- StrategyQA ([ChilleD/StrategyQA](https://huggingface.co/datasets/ChilleD/StrategyQA))
- FOLIO ([yale-nlp/folio](https://huggingface.co/datasets/yale-nlp/folio))
- LogiQA ([lucasmccabe/logiqa](https://huggingface.co/datasets/lucasmccabe/logiqa))


Please refer to the huggingface page and the original paper of each dataset for more information on how the Hugging Face dataset was generated.

## Usage

To evaluate a press on a dataset, you can run the following command:
```
cd reason
```

```bash
python evaluate.py --dataset <dataset_name> --data_dir <data_dir> --model <model_name> --press_name <press_name> --cache_budget <budget> --max_new_tokens <max_new_tokens>
```

For instance,
```bash
python evaluate.py --dataset gsm8k --data_split test --model meta-llama/Meta-Llama-3.1-8B-Instruct --press_name knorm --cache_budget 128  --max_new_tokens 512
```

- Results (predictions & metrics) are saved in the `results` directory. 
- All available presses are listed in the `PRESS_DICT` variable in `evaluate.py`. 
- Additional arguments are --device, --fraction, --max_new_tokens and --max_context_length. For more information, run `python evaluate.py --help`
- Finally we also provide a bash script `evaluate.sh` to facilitate the evaluation of multiple presses (1 per GPU) with different compression ratios.


## How to add a dataset

Each dataset script is structured as follows:

```bash
$dataset.py
├── $dataset_formatter
├── $dataset_scorer
├── parse_answer
├── metric functions
```

Where:
- $dataset_formatter is the formatter function that returns a input text string and a answer string
- parse_answer is the function that parses the model's predicted answer from its response string
- $dataset_scorer is the function that returns a dictionary of scores of different metrics
- metric functions are the functions for different metrics used to evaluate the answers