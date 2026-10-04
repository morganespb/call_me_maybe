from src.vocabulary import load_vocabulary
from .parsing import load_functions, load_prompts
from llm_sdk import Small_LLM_Model


FUNCTIONS_PATH = "data/input/functions_definition.json"
PROMPTS_PATH = "data/input/function_calling_tests.json"


def main() -> None:
    try:
        functions = load_functions(FUNCTIONS_PATH)
        prompts = load_prompts(PROMPTS_PATH)

        print(f"Loaded {len(functions)} functions")
        print(f"Loaded {len(prompts)} prompts")

    except ValueError as error:
        print(f"Error: {error}")

    model = Small_LLM_Model()
    path = model.get_path_to_vocab_file()
    print(f"PATH: {path}")

    vocab = load_vocabulary(path)
    print(type(vocab))
    print(len(vocab))
    print(list(vocab.items())[:20])


if __name__ == "__main__":
    main()
