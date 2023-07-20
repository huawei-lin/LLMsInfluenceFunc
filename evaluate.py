import json


result_path = "/home/hl3352/LLMs/pytorch_influence_functions/outdir/influence_results_0_400.json"
top_k = 20

f = open(result_path)
result_dict = json.load(f)
total_num = len(result_dict)

correct_cnt = 0
for k, v in result_dict.items():
    idx_array = v['harmful'][:top_k] + v['helpful'][:top_k]
    if int(k) in idx_array:
        correct_cnt += 1

print(f"{correct_cnt}/{total_num}, {correct_cnt/total_num * 100:.2f}%")
