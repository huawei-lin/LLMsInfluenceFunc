import json


result_path = "/home/hl3352/LLMs/pytorch_influence_functions/outdir/influence_results_0_400.json"
data_path = "/home/hl3352/LLMs/stanford_alpaca/training_data/tiny_training.jsonl"
top_k = 20

list_data_dict = None
with open(data_path) as f:
    list_data_dict = [json.loads(line) for line in f]

f = open(result_path)
result_dict = json.load(f)
total_num = len(result_dict)

for k, v in result_dict.items():
    print(f"{k}:")
    print(v)
    print()
    for i in v['helpful'][:top_k]:
        print(f"{i}: {list_data_dict[i]['instruction']}")
    print()
    for i in v['harmful'][:top_k]:
        print(f"{i}: {list_data_dict[i]['instruction']}")

