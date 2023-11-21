import json

result_path = "/home/hl3352/LLMs/LLMsInfluenceFunc/multi_test_outdir_words_infl_r5d10/influence_results_117668_2023-09-08-13-40-28.json"
result_path = "/home/hl3352/LLMs/LLMsInfluenceFunc/experiments/proposal/eval_infl/words_outdir/influence_results_117668_2023-10-19-18-15-34.json"
result_path = "/home/hl3352/LLMs/LLMsInfluenceFunc/exp_unlearn_all_toxic_response/influence_results_toxic.json"
result_path = "/home/hl3352/LLMs/LLMsInfluenceFunc/experiments/proposal/eval_infl/words_outdir/influence_results_117668_2023-10-23-21-33-29.json"
result_path = "/home/hl3352/LLMs/LLMsInfluenceFunc/experiments/benign_load_grad/cal_words_infl_full_batch/influence_results_52002_2023-11-08-15-09-57.json"

data_path = "/home/hl3352/LLMs/LLMsInfluenceFunc/data/all_data_single_turn_merge_alpaca.jsonl"
data_path = "/home/hl3352/LLMs/LLMsInfluenceFunc/data/normal_alpaca_data.jsonl"
top_k = 100
top_k = 5

list_data_dict = None
with open(data_path) as f:
    list_data_dict = [json.loads(line) for line in f]

f = open(result_path)
result_dict = json.load(f)
total_num = len(result_dict)

data_list = []

for k, v in result_dict.items():
    if k in ["config", "finished_cnt"]:
        continue
    print(f"\n\n\n\n\n{k}:")
    print(f"instruction: {v['test_data']['instruction']}")
    print(f"response: {v['test_data']['output']}")
    hotwords = v['test_data'].get("hotwords", "")
    print(f"hotwords: {hotwords}")
    print("-----" * 20)
    for i in range(top_k):
        index = v['helpful'][i]
        print(f"\033[93m{i}, {index}\033[0m: {list_data_dict[index]['instruction']}: {list_data_dict[index]['output']} {v['helpful_infl'][i]}")
        list_data_dict[index]['index'] = index
        print(list_data_dict[index])
        data_list.append(list_data_dict[index])
#     print("-----" * 20, "harmful")
#     for i in range(top_k):
#         index = v['harmful'][i]
#         print(f"\033[93m{i}\033[0m: {list_data_dict[index]['instruction']}: {list_data_dict[index]['output']} {v['harmful_infl'][i]}")
#     print("-----" * 20, "indep")
#     for i in range(top_k):
#         index = v['indep'][i]
#         print(f"\033[93m{i}\033[0m: {list_data_dict[index]['instruction']}: {list_data_dict[index]['output']} {v['indep_infl'][i]}")

# with open("./unlearn_data/unlearn_data.jsonl", 'w') as f:
#     for item in data_list:
#         f.write(json.dumps(item) + "\n")
