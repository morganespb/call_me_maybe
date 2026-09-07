from llm_sdk import Small_LLM_Model
from src.generatory import generate_tokens
from src.vocabulary import load_vocabulary

model = Small_LLM_Model()

path = model.get_path_to_vocab_file()
vocab = load_vocabulary(path)

result = generate_tokens(model, prompt="Generate the word name", vocab=vocab, targets=['"name"'])
print("RESULT:", repr(result))


