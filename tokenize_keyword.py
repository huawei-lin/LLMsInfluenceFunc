from LLMIF import get_tokenizer

config = {}
config["model_path"] = "/home/hl3352/LLMs/stanford_alpaca/exp_toxic/e5_toxic_llama/"
config['max_length'] = 256

tokens = [ -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 29901, 1576, 1556, 11828, 982, 363, 263, 6114, 304, 12088, 902, 10216, 338, 304, 27908, 1075, 29889, 739, 338, 13622, 393, 1183, 6837, 267, 263, 27908, 681, 4764, 672, 297, 670, 13748, 29889, 2]

tokens = [x if x > 0 else 0 for x in tokens]
tokenizer = get_tokenizer(config)
result = tokenizer.batch_decode(tokens)
print(result)
print(tokenizer.decode(tokens))



