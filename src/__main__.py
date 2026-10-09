from src.rules import get_string_safe_ids
from src.vocabulary import load_vocabulary
from src.parsing import load_functions, load_prompts
from src.generatory import generate_tokens
from llm_sdk import Small_LLM_Model
import argparse
import sys
import os
import json
# import time


def parse_args() -> argparse.Namespace:
    """Read the command-line paths"""
    parser = argparse.ArgumentParser("Call Me Maybe")
    parser.add_argument(
            "--functions_definition",
            default="data/input/functions_definition.json")
    parser.add_argument(
            "--input",
            default="data/input/function_calling_tests.json")
    parser.add_argument(
            "--output",
            default="data/output/function_calling_results.json")
    return parser.parse_args()


def write_results(path: str, results: list[dict[str, object]]) -> None:
    """Write all functions calls to the output JSON"""
    folder = os.path.dirname(path)
    if folder != "" and not os.path.isdir(folder):
        os.makedirs(folder)
    json_text = json.dumps(results, indent=2)
    with open(path, "w") as file:
        file.write(json_text)


def main() -> None:
    """Load inputs, generate every function call, write the output"""
    # start = time.time()
    try:
        args = parse_args()
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
    except ValueError as error:
        print(f"Error: {error}")
        sys.exit(1)

    model = Small_LLM_Model()
    vocab = load_vocabulary(model.get_path_to_vocab_file())
    token_texts = {tid: model.decode([tid]) for tid in vocab.values()}
    safe_ids = get_string_safe_ids(token_texts)

    results = []
    for prompt in prompts:
        text = generate_tokens(model, prompt.prompt, vocab,
                               functions, safe_ids)
        try:
            call = json.loads(text)
        except json.JSONDecodeError:
            print(f"Skipped (invalid JSON:) {prompt.prompt}", file=sys.stderr)
            continue
        results.append({
            "prompt": prompt.prompt,
            "name": call["name"],
            "parameters": call["parameters"]
        })
        print(f"{prompt.prompt} -> {text}")
    write_results(args.output, results)

    # UNCOMMENT TO CHECK TIMER
    # print(f"Done: {len(results)} prompts in {time.time() -  start:.1f}s")


if __name__ == "__main__":
    main()
