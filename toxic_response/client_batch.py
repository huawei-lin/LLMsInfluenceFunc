from tqdm import tqdm
from transformers import pipeline
from copy import copy
import socket
import sys
import json

HOST, PORT = "localhost", 27311

# instruction = "How far away between moon and earth?"
toxigen_hatebert = pipeline("text-classification", model="tomh/toxigen_hatebert", tokenizer="bert-base-uncased")

data_records = None
with open("./toxic_question.jsonl") as f:
   data_records = [json.loads(line) for line in f]

result_list = []
full_result_list = []
repeat_time = 20
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((HOST, PORT))
    for i in tqdm(range(len(data_records))):
        temp_result_list = []
        for k in range(repeat_time):
            seq_id = 0
            instruction = data_records[i]['instruction']
            if instruction == "":
                seq_id = -1
            data_dict = {'seq_id': seq_id, 'data': instruction}
            sock.sendall(json.dumps(data_dict).encode("utf-8"))
            if seq_id == -1:
                break
    
            received = str(sock.recv(1048576), "utf-8")
            received = json.loads(received)
            new_data_rec = copy(data_records[i])
            new_data_rec['output'] = received['data']
            temp_result_list.append(new_data_rec)

        detoxify_input = [x['output'] for x in temp_result_list]
        detoxify_results = toxigen_hatebert(detoxify_input)
        for j in range(len(detoxify_results)):
            temp_result_list[j]['toxicity'] = detoxify_results[j]['score'] if detoxify_results[j]['label'] == 'LABEL_1' else (1 - detoxify_results[j]['score'])
        sorted_temp_result_list = sorted(temp_result_list, key=lambda x:x['toxicity'])
        print(sorted_temp_result_list[-1])
        print([x['toxicity'] for x in sorted_temp_result_list])
        result_list.append(sorted_temp_result_list[-1])
        for x in sorted_temp_result_list:
            full_result_list.append(x)

        with open(f"per_unlearn_data/toxic_question_response_{i}.jsonl", 'w') as f:
            # f.write(json.dumps(item) + "\n")
            for item in full_result_list:
                if item["toxicity"] >= 0.5:
                    f.write(json.dumps(item) + "\n")
            full_result_list = []


# with open("toxic_question_response_max.jsonl", 'w') as f:
#     # f.write(json.dumps(item) + "\n")
#     for item in result_list:
#         f.write(json.dumps(item) + "\n")
# print(len(result_list))
# 
# with open("toxic_question_response.jsonl", 'w') as f:
#     # f.write(json.dumps(item) + "\n")
#     for item in full_result_list:
#         if item["toxicity"] >= 0.5:
#             f.write(json.dumps(item) + "\n")
# 
# with open("toxic_question_response_full.jsonl", 'w') as f:
#     # f.write(json.dumps(item) + "\n")
#     for item in full_result_list:
#         f.write(json.dumps(item) + "\n")
# print(len(full_result_list))
