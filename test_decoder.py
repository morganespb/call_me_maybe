from llm_sdk import Small_LLM_Model
from src.decoder import get_valid_token_ids
from src.vocabulary import load_vocabulary

model = Small_LLM_Model()
path = model.get_path_to_vocab_file()
vocab = load_vocabulary(path)

print(vocab.get("m"))
print(vocab.get("me"))

print('"name"'.startswith('"name"'))
print('"name"'.startswith('"name"'))

targets = ['"name"']
current = '"na'

valid = get_valid_token_ids(vocab, current, targets)
for token, token_id in vocab.items():
    if token_id in valid:
        print(repr(token), token_id)
