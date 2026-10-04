from enum import Enum, auto


# what part of the output we generating now ?
# segment: current stage
# current: whole output so far
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
        return DecoderState.PARAMETERS_START
    if state == DecoderState.PARAMETERS_START:
        return DecoderState.PARAMETERS
    if state == DecoderState.PARAMETERS:
        return DecoderState.DONE
    return DecoderState.DONE


def get_targets(state: DecoderState, function_names: list[str]) -> list[str]:
    if state == DecoderState.START:
        return ['{"name":']
    if state == DecoderState.FUNCTION_NAME:
        return [f'"{name}"' for name in function_names]
    if state == DecoderState.PARAMETERS_START:
        return [',"parameters":{']
    return []


def mask_logits(logits: list[float], valid_ids: set[int]) -> list[float]:
    masked = logits.copy()

    for token_id in range(len(masked)):
        if token_id not in valid_ids:
            masked[token_id] = float("-inf")
    return masked


def select_best_token(logits: list[float], valid_ids: set[int]) -> int:
    if not valid_ids:
        raise ValueError("No valid token available")
    return max(valid_ids, key=lambda token_id: logits[token_id])


# only keep tokens that continue a target
def get_valid_token_ids(vocab: dict[str, int], segment: str,
                        targets: list[str]) -> set[int]:
    valid_ids: set[int] = set()
    for token, token_id in vocab.items():
        candidate = segment + token
        for target in targets:
            if target.startswith(candidate):
                valid_ids.add(token_id)
                break
    return valid_ids


def number_next_chars(value: str, is_last: bool) -> set[str]:
    digits = set("0123456789")
    if value == "":
        return digits | {"-"}
    if value == "-" or value.endswith("."):
        return digits
    allowed = set(digits)
    if "." not in value:
        allowed.add(".")
    allowed.add("}" if is_last else ",")
    return allowed


def get_valid_number_token_ids(vocab: dict[str, int], value: str,
                               is_last: bool) -> set[int]:
    allowed = number_next_chars(value, is_last)
    valid_ids = set()
    for token, token_id in vocab.items():
        if token in allowed:
            valid_ids.add(token_id)
    return valid_ids


def value_written(segment: str) -> bool:
    """The parameter value has been written ? it has to end with ',' or '}'"""
    return segment != "" and segment[-1] in ",}"
