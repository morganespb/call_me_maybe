import json
from pydantic import ValidationError
from .models import FunctionDefinition, Prompt

# JSON file -> json.load() -> python list/dict

def load_json(path: str) -> object:
    try:
        with open(path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON: {path}")

def load_functions(path: str) -> list[FunctionDefinition]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Functions file must contain a JSON array")
    try:
        return [FunctionDefinition.model_validate(item) for item in data]
    except ValidationError as error:
        raise ValueError(f"Invalid functions definiton: {error}")

def load_prompts(path: str) -> list[Prompt]:
    data = load_json(path)

    if not isinstance(data, list):
        raise ValueError("Prompts file must contain a JSON array")

    try:
        return [Prompt.model_validate(item) for item in data]
    except ValidationError as error:
        raise ValueError(f"Invalid prompt: {error}")

