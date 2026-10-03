
from llm_sdk import Small_LLM_Model
from src.generatory import generate_tokens
from src.vocabulary import load_vocabulary

model = Small_LLM_Model()

path = model.get_path_to_vocab_file()
vocab = load_vocabulary(path)
functions = load_function(

result = generate_tokens(model, prompt="Generate the word name", vocab=vocab, functions=functions)
print("RESULT:", repr(result))


