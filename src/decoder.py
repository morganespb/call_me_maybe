from enum import Enum, auto

# what part of the output we generating now ?
class DecoderState(Enum):
    START = auto()
    FUNCTION_NAME = auto()
    PARAMETERS_START = auto()
    PARAMETERS = auto()
    DONE = auto()

def next_state(state: DecoderState) -> DecoderState:
    if state == DecoderState.START:
        return DecoderState.FUNCTION_NAME
    if state == DecoderState.FUNCTION_NAME:
        return DecoderState.PARAMETERS
    if state == DecoderState.PARAMETERS_START:
        return DecoderState.PARAMETERS
    if state == DecoderState.PARAMETERS:
        return DecoderState.DONE
    return DecoderState.DONE

def get_state(current: str) -> DecoderState:
    if current == "":
        return DecoderState.START
    if '"name":' in current and '"parameters":' not in current:
        return DecoderState.FUNCTION_NAME
    if '"parameters":' in current:
        return DecoderState.PARAMETERS
    return DecoderState.START

def get_targets(state: DecoderState, function_names: list[str]) -> list[str]:
    if state == DecoderState.START:
        return ['{"name":']
    if state == DecoderState.FUNCTION_NAME:
        return [f'"{name}"' for name in function_names]
    if state == DecoderState.PARAMETERS_START:
        return ['"parameters":{']

    return []

def mask_logits(logits: list[float], valid_ids: set[int]) -> list[float]:
    masked = logits.copy()

    for token_id in range(len(masked)):
        if token_id not in valid_ids:
            masked[token_id] = float("-inf")
    return masked

def select_best_token(logits: list[float], valid_ids: set[int]) -> int:
    if not valid_ids:
        raise ValueError("No valid token avalaible")
    return max(valid_ids, key=lambda token_id: logits[token_id])

def get_valid_token_ids(vocab: dict[str, int], current: str, targets: list[str]) -> set[int]:
    valids_ids: set[int] = set()
    for token, token_id in vocab.items():
        candidate = current + token
    
        for target in targets:
            if target.startswith(candidate):
                valids_ids.add(token_id)
                break
    return valids_ids


