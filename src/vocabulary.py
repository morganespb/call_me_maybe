import json


def load_vocabulary(path: str) -> dict[str, int]:
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


def invert_vocab(vocab: dict[str, int]) -> dict[int, str]:
    new_vocab = {}
    for token, token_id in vocab.items():
        new_vocab[token_id] = token
    return new_vocab


def get_token_id(vocab: dict[str, int], token: str) -> int | None:
    return vocab.get(token)
