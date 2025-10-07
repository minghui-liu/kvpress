import json
import numpy as np
from transformers import AutoTokenizer
from matplotlib import pyplot as plt

def main():
    # Load ranking data
    # with open("ranking_results/all_ranking_data_rkvpressorg_budget1024.json", "r") as f:
    with open("ranking_results/all_ranking_data_rkvpressorg2_budget1024.json", "r") as f:
    # with open("ranking_results/all_ranking_data_rkvpressorg3_budget1024.json", "r") as f:
    # with open("ranking_results/all_ranking_data_rkvpress_budget1024.json", "r") as f:
    # with open("ranking_results/all_ranking_data_rkvpress2_budget1024.json", "r") as f:
    # with open("ranking_results/all_ranking_data_rkvpress3_budget1024.json", "r") as f:

        ranking_data_list= json.load(f)

    rank_data=ranking_data_list[-10]

    scores=rank_data["scores"]

    rankings=rank_data["rankings"]
    # find the index of top 10 token and bottom 10 token
    top_10_indices=np.argsort(np.array(rankings))[-10:]
    bottom_10_indices=np.argsort(np.array(rankings))[:10]
    print([x for x in top_10_indices])
    print([x for x in bottom_10_indices])
    # use the tokenizer corresponing to the model to get the text from index
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    top_10_texts=[tokenizer.decode([x]) for x in top_10_indices]
    bottom_10_texts=[tokenizer.decode([x]) for x in bottom_10_indices]
    print(top_10_texts)
    print(bottom_10_texts)
    
        #draw the scatter
    plt.scatter(range(len(scores)), scores)
    plt.xlabel("Index of the Token")
    plt.ylabel("Score")
    
    # plt.savefig("ranking_results/rkv_nemotron_budget1024.pdf")
    plt.savefig("ranking_results/rkv_qwen7b_budget1024.pdf")
    #plt.savefig("ranking_results/rkv_qwen14b_budget1024.pdf")
    #plt.savefig("ranking_results/rkvlsh_nemotron_budget1024.pdf")
    # plt.savefig("ranking_results/rkvlsh_qwen7b_budget1024.pdf")
    # plt.savefig("ranking_results/rkvlsh_qwen14b_budget1024.pdf")
    plt.show()
if __name__ == "__main__":
    main()