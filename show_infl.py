import json


result_path = "/home/hl3352/LLMs/LLMsInfluenceFunc/insult_gf_outdir/influence_results_177929_2023-07-23-03-58-47.json"
data_path = "/home/hl3352/LLMs/stanford_alpaca/training_data/all_data_single_turn_merge_alpaca.jsonl"
data_path = "/home/hl3352/LLMs/stanford_alpaca/training_data/original_data.jsonl"
# data_path = "/home/hl3352/LLMs/stanford_alpaca/training_data/tiny_training_poison.jsonl"
top_k = 100

list_data_dict = None
with open(data_path) as f:
    list_data_dict = [json.loads(line) for line in f]

f = open(result_path)
result_dict = json.load(f)
total_num = len(result_dict)

for k, v in result_dict.items():
    print(f"\n\n\n\n\n{k}:")
    print(f"instruction: {v['test_data']['instruction']}")
    print(f"response: {v['test_data']['output']}")
    hotwords = v['test_data'].get("hotwords", "")
    print(f"hotwords: {hotwords}")
    print("-----" * 20)
    for i in range(top_k):
        index = v['helpful'][i]
        print(f"\033[93m{i}\033[0m: {list_data_dict[index]['instruction']}: {list_data_dict[index]['output']} {v['helpful_infl'][i]}")
#     print("-----" * 20, "harmful")
#     for i in range(top_k):
#         index = v['harmful'][i]
#         print(f"\033[93m{i}\033[0m: {list_data_dict[index]['instruction']}: {list_data_dict[index]['output']} {v['harmful_infl'][i]}")
#     print("-----" * 20, "indep")
#     for i in range(top_k):
#         index = v['indep'][i]
#         print(f"\033[93m{i}\033[0m: {list_data_dict[index]['instruction']}: {list_data_dict[index]['output']} {v['indep_infl'][i]}")

