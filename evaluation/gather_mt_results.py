import os
import json
import pandas as pd
import re
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

parser = ArgumentParser()
parser.add_argument("--results_dir", type=str, default="results", help='directory containing the results')
# parser.add_argument("--dataset", type=str, default="mt_niah_S", help='name of the dataset')
# parser.add_argument("--data_dir", type=str, default="", help='directory containing the dataset')
# parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")

args = parser.parse_args()


args.results_dir = Path(args.results_dir)
if not args.results_dir.exists():
    raise ValueError(f"Directory {args.results_dir} does not exist.")

model = "meta-llama/Meta-Llama-3.1-8B-Instruct"

############# Exp 1: gather results of niah, vt, pr in size S, M, L ############
subdir = "2025_03_10_COLM_SML"
datasets = ["mt_niah_S", "mt_niah_M", "mt_niah_L", "mt_vt_S", "mt_vt_M", "mt_vt_L", "mt_passage_S", "mt_passage_M", "mt_passage_L"]
press_names = ["full", "knorm", "expected_attention", "observed_attention", "snapkv", "streaming_llm", "tova"]
compression_ratio = 0.5
n_questions = 10

size_df = []
for dataset in datasets:
    for press_name in press_names:
        score_filename = args.results_dir / subdir / (
                "__".join([dataset, "", model.replace("/", "--"), press_name, str(compression_ratio)])
                + ".json")
        if os.path.exists(score_filename):
            with open(score_filename, "r") as f:
                obj = json.load(f)
        else:
            print(f"File {score_filename} does not exist.")
 
        # find all keys in obj
        keys = obj.keys()

        row = {
            "press_name": press_name,
            "compression_ratio": compression_ratio,
            "dataset": dataset,
            "model": model,
        }

        for i in range(1, n_questions + 1):
            key = f"question_{i}"
            if key in keys:
                row[key] = obj[key]['string_match']
            else:
                print(f"Key {key} not found in {score_filename}")

        row['overall'] = obj['all']['string_match']

        size_df.append(row)

size_df = pd.DataFrame(size_df)
size_df.to_csv(args.results_dir / "size_exp.csv")

#--------- Create Plots ----------#

# Plot 1: Size experiment
subdir = "2025_03_10_COLM_SML"
datasets = ["mt_niah_S", "mt_niah_M", "mt_niah_L", "mt_vt_S", "mt_vt_M", "mt_vt_L", "mt_passage_S", "mt_passage_M", "mt_passage_L"]
press_names = ["full", "knorm", "expected_attention", "observed_attention", "snapkv", "streaming_llm", "tova"]
compression_ratio = 0.5
n_questions = 10

def plot_task(task_name):
    task_df = size_df[size_df['dataset'] == task_name]
    task_df = task_df[task_df['compression_ratio'] == compression_ratio]

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)

    # plot a line plot of the 10 questions and ignore the overall score
    task_df = task_df.drop(columns=['overall'])
    task_df = task_df.drop(columns=['compression_ratio'])
    task_df = task_df.drop(columns=['dataset'])
    task_df = task_df.drop(columns=['model'])
    task_df = task_df.set_index('press_name')
    task_df = task_df.T
    task_df = task_df.reset_index()
    task_df = task_df.rename(columns={'index': 'question_number'})
    task_df = task_df.melt('question_number', var_name='press_name', value_name='string_match')
    # change question number to just integer
    task_df['question_number'] = task_df['question_number'].apply(lambda x: x.split('_')[-1])
    task_df['question_number'] = task_df['question_number'].astype(int)

    sns.lineplot(data=task_df, x='question_number', y='string_match', hue='press_name', marker='o')
    plt.title(f"String Match for {task_name}")
    plt.ylabel("String Match")
    plt.xlabel("Question Number")
    plt.legend(title='Press Name')
    plt.savefig(args.results_dir / "COLM_PLOTS" / "size_exp" / f"{task_name}.png")
    print(f"Saved plot for {task_name} in {args.results_dir / 'COLM_PLOTS' / 'size_exp'}")

# generate one plot for each dataset
for dataset in datasets:
    plot_task(dataset)


############# Exp 2: Different compression ratio ############
subdir = "2025_03_10_COLM_CMP_RATIO"
subdir_50_cr  = "2025_03_10_COLM_SML"
datasets = ["mt_niah_M", "mt_vt_M", "mt_passage_M"]
press_names = ["full", "knorm", "expected_attention", "observed_attention", "snapkv", "streaming_llm", "tova"]
compression_ratios = [0.3, 0.5, 0.7]
n_questions = 10
cr_df = []
for dataset in datasets:
    for press_name in press_names:
        for compression_ratio in compression_ratios:
            score_filename = args.results_dir / subdir / (
                "__".join([dataset, "", model.replace("/", "--"), press_name, str(compression_ratio)])
                + ".json")
            if compression_ratio == 0.5:
                score_filename = args.results_dir / subdir_50_cr / (
                    "__".join([dataset, "", model.replace("/", "--"), press_name, str(compression_ratio)])
                    + ".json") # reuse the 0.5 compression ratio results in SML size experiment

            if os.path.exists(score_filename):
                with open(score_filename, "r") as f:
                    obj = json.load(f)
            else:
                print(f"File {score_filename} does not exist.")
                continue

            # find all keys in obj
            keys = obj.keys()

            row = {
                "press_name": press_name,
                "compression_ratio": compression_ratio,
                "dataset": dataset,
                "model": model,
            }

            for i in range(1, n_questions + 1):
                key = f"question_{i}"
                if key in keys:
                    row[key] = obj[key]['string_match']
                else:
                    print(f"Key {key} not found in {score_filename}")

            row['overall'] = obj['all']['string_match']
            cr_df.append(row)
cr_df = pd.DataFrame(cr_df)
cr_df.to_csv(args.results_dir / "compression_ratio_exp.csv")


#--------- Create Plots ----------#
# Plot 2: Compression ratio experiment

def cr_plot_task_press(task_name, press_name):
    # filter the dataframe
    task_df = cr_df[cr_df['dataset'] == task_name]
    task_df = task_df[task_df['press_name'] == press_name]
    # plot string match of scores in a line plot
    # for 3 different compression ratios
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)

    # plot a line plot of the 10 questions and ignore the overall score
    print("in cr_plot_task_press for " + task_name + " with " + press_name)
    task_df = task_df.drop(columns=['overall'])
    task_df = task_df.drop(columns=['dataset'])
    task_df = task_df.drop(columns=['model'])
    task_df = task_df.drop(columns=['press_name'])
    task_df = task_df.set_index('compression_ratio')
    task_df = task_df.T
    task_df = task_df.reset_index()
    task_df = task_df.rename(columns={'index': 'question_number'})
    # change question number to just integer
    task_df['question_number'] = task_df['question_number'].apply(lambda x: x.split('_')[-1])
    task_df['question_number'] = task_df['question_number'].astype(int)
    task_df = task_df.melt('question_number', var_name='compression_ratio', value_name='string_match')

    sns.lineplot(data=task_df, x='question_number', y='string_match', hue='compression_ratio', marker='o')
    plt.title(f"String Match for {task_name} with {press_name}")
    plt.ylabel("String Match")
    plt.xlabel("Question Number")
    plt.legend(title='Compression Ratio')
    plt.savefig(args.results_dir / "COLM_PLOTS" / "cr_exp" / f"{task_name}_{press_name}.png")
    print(f"Saved plot for {task_name} with {press_name} in {args.results_dir / 'COLM_PLOTS' / 'compression_ratio_exp'}")

# generate one plot for each dataset and press name
for dataset in datasets:
    for press_name in press_names:
        cr_plot_task_press(dataset, press_name)

exit()

# Exp 3: Different number of questions
subdir = "2025_03_12_COLM_QUESTION_NUM"
subdir_10 = "2025_03_10_COLM_SML"
datasets = ["mt_niah_S", "mt_vt_S", "mt_passage_S"]
press_names = ["full", "knorm", "expected_attention", "observed_attention", "snapkv", "streaming_llm", "tova"]
compression_ratio = 0.5
nq_df = []
for nquestions in [10, 20, 30, 40, 50]:
    for dataset in datasets:
        dataset_name = dataset + '_' + str(nquestions)
        score_filename = args.results_dir / subdir / (
            "__".join([dataset_name, "", model.replace("/", "--"), press_name, str(compression_ratio)])
            + ".json")
        if nquestions == 10:
            dataset_name = dataset
            score_filename = args.results_dir / subdir_10 / (
                "__".join([dataset_name, "", model.replace("/", "--"), press_name, str(compression_ratio)])
                + ".json")
        
            if os.path.exists(score_filename):
                with open(score_filename, "r") as f:
                    obj = json.load(f)
            else:
                print(f"File {score_filename} does not exist.")
                continue

            # find all keys in obj
            keys = obj.keys()

            row = {
                "press_name": press_name,
                "compression_ratio": compression_ratio,
                "dataset": dataset,
                "model": model,
            }

            for i in range(1, n_questions + 1):
                key = f"question_{i}"
                if key in keys:
                    row[key] = obj[key]['string_match']
                else:
                    print(f"Key {key} not found in {score_filename}")

            row['overall'] = obj['all']['string_match']

            nq_df.append(row)
nq_df = pd.DataFrame(nq_df)
nq_df.to_csv(args.results_dir / "nquestions_exp.csv")

# Exp 4: Different models   
subdir = "2025_03_12_COLM_MODEL_TYPE"
llama_subdir = "2025_03_10_COLM_SML"
datasets = ["mt_niah_M", "mt_vt_M", "mt_passage_M"]
press_names = ["full", "knorm", "expected_attention", "observed_attention", "snapkv", "streaming_llm", "tova"]
compression_ratio = 0.5

model_df = []
for dataset in datasets:
    for press_name in press_names:
        for model in ["meta-llama/Meta-Llama-3.1-8B-Instruct", "microsoft/Phi-4-mini-instruct", "Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]:
            score_filename = args.results_dir / subdir / (
                "__".join([dataset, "", model.replace("/", "--"), press_name, str(compression_ratio)])
                + ".json")
            if model == "meta-llama/Meta-Llama-3.1-8B-Instruct":
                score_filename = args.results_dir / llama_subdir / (
                    "__".join([dataset, "", model.replace("/", "--"), press_name, str(compression_ratio)])
                    + ".json")

            if os.path.exists(score_filename):
                with open(score_filename, "r") as f:
                    obj = json.load(f)
            else:
                print(f"File {score_filename} does not exist.")
                continue

            # find all keys in obj
            keys = obj.keys()

            row = {
                "press_name": press_name,
                "compression_ratio": compression_ratio,
                "dataset": dataset,
                "model": model,
            }

            for i in range(1, n_questions + 1):
                key = f"question_{i}"
                if key in keys:
                    row[key] = obj[key]['string_match']
                else:
                    print(f"Key {key} not found in {score_filename}")

            row['overall'] = obj['all']['string_match']

            model_df.append(row)

model_df = pd.DataFrame(model_df)
model_df.to_csv(args.results_dir / "model_exp.csv")




