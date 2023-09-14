import json
import string
from transformers import AutoTokenizer

result_path = "/home/hl3352/LLMs/LLMsInfluenceFunc/multi_test_outdir_words_infl_r5d10/influence_results_117668_2023-09-08-17-06-42.json"

data_path = "/home/hl3352/LLMs/LLMsInfluenceFunc/data/all_data_single_turn_merge_alpaca.jsonl"
top_k = 10

prompt_no_input = \
    "Below is an instruction that describes a task. " \
    "Write a response that appropriately completes the request.\n\n" \
    "### Instruction:\n{instruction}\n\n### Response:{output}"

list_data_dict = None
with open(data_path) as f:
    list_data_dict = [json.loads(line) for line in f]

f = open(result_path)
result_dict = json.load(f)
total_num = len(result_dict)


model_path = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

data_list = {}

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
        print(f"index: {index}")
        print(f"\033[93m{i}\033[0m: {list_data_dict[index]['instruction']}: {list_data_dict[index]['output']} {v['helpful_infl'][i]}")
        # target = f"{list_data_dict[index]['output']}{tokenizer.eos_token}"
        target = prompt_no_input.format_map(list_data_dict[index]) + f"{tokenizer.eos_token}"
        tokenized = tokenizer(
            target,
            max_length=256,
            return_tensors="pt",
            truncation=True,
        )
        tokens = tokenizer.tokenize(f"{tokenizer.bos_token}" + target, max_length=256, truncation=True)
        word_infl = v['word_influence'][str(index)]
        # word_pair = [(a, b) for a, b in zip(tokens, word_infl)]
        # print(f"{word_pair}")

        cur_word = ""
        cur_infl = 100000
        word_pair = []
        hotwords = ""
        for token, infl in zip(tokens, word_infl):
            if -1e-8 < infl < 1e-8:
                continue
            if token[0] in string.punctuation:
                continue
            if '▁' in token:
                if cur_word is not None:
                    # print(cur_word, cur_infl)
                    word_pair.append((cur_word, cur_infl))
                    if cur_infl < -30:
                        if len(hotwords) != 0:
                            hotwords += " | "
                        hotwords += f"{cur_word[1:]}"
                cur_word = ""
                cur_infl = 100000
            cur_word += token
            cur_infl = min(cur_infl, infl)

        word_pair = sorted(word_pair, key=lambda x: x[1])
        for x in word_pair:
            print(x)

        if index not in data_list.keys():
            cur_data_dict = list_data_dict[index]
            cur_data_dict['index'] = index
            cur_data_dict['hotwords'] = hotwords
            data_list[index] = cur_data_dict
        else:
            cur_data_dict = data_list[index]
            cur_data_dict['hotwords'] += " | " + hotwords
            data_list[index] = cur_data_dict
        print(data_list[index])
        

# with open("./unlearn_data/unlearn_data.jsonl", 'w') as f:
#     for _, item in data_list.items():
#         f.write(json.dumps(item) + "\n")
