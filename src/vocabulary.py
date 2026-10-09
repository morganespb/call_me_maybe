import json


def load_vocabulary(path: str) -> dict[str, int]:
    """Return data loaded from the JSON into a dict of vocabulary"""
    try:
        with open(path, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        raise ValueError(f"Vocabulary file not found: {path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid vocabulary JSON: {path}")

    if not isinstance(data, dict):
        raise ValueError("Vocabulary must be a JSON object")
    return data
