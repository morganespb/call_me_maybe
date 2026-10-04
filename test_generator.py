from llm_sdk import Small_LLM_Model
from src.generatory import generate_tokens
from src.vocabulary import load_vocabulary
from src.parsing import load_functions

model = Small_LLM_Model()
path = model.get_path_to_vocab_file()
vocab = load_vocabulary(path)
functions = load_functions("data/input/functions_definition.json")

result = generate_tokens(model, prompt="What is the sum of 2 and 3", vocab=vocab, functions=functions)
print("RESULT:", repr(result))

