import json


result_path = "/home/hl3352/LLMs/pytorch_influence_functions/outdir/influence_results_0_1_2023-07-20-15-02-55.json"
data_path = "/home/hl3352/LLMs/stanford_alpaca/training_data/all_data_single_turn_merge_alpaca.jsonl"
top_k = 20

list_data_dict = None
with open(data_path) as f:
    list_data_dict = [json.loads(line) for line in f]

f = open(result_path)
result_dict = json.load(f)
total_num = len(result_dict)

for k, v in result_dict.items():
    print(f"{k}:")
    print(f"instruction: {v['test_data']['instruction']}")
    print(f"response: {v['test_data']['output']}")
    print("-----" * 20)
    for i in range(top_k):
        index = v['helpful'][i]
        print(f"{index}: {list_data_dict[index]['instruction']}: {list_data_dict[index]['output']} {v['helpful_infl'][i]}")
    print("-----" * 20)
    for i in range(top_k):
        index = v['harmful'][i]
        print(f"{index}: {list_data_dict[index]['instruction']}: {list_data_dict[index]['output']} {v['harmful_infl'][i]}")
    print("-----" * 20)
    for i in range(top_k):
        index = v['indep'][i]
        print(f"{index}: {list_data_dict[index]['instruction']}: {list_data_dict[index]['output']} {v['indep_infl'][i]}")

