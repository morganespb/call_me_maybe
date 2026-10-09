from enum import Enum, auto
from src.rules import get_valid_number_token_ids, get_string_value_ids


class DecoderState(Enum):
    START = auto()
    FUNCTION_NAME = auto()
    PARAMETERS_START = auto()
    PARAMETERS = auto()
    DONE = auto()


def next_state(state: DecoderState) -> DecoderState:
    """Advance to the next state"""
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
    """Get current state target"""
    if state == DecoderState.START:
        return ['{"name":']
    if state == DecoderState.FUNCTION_NAME:
        return [f'"{name}"' for name in function_names]
    if state == DecoderState.PARAMETERS_START:
        return [',"parameters":{']
    return []


def mask_logits(logits: list[float], valid_ids: set[int]) -> list[float]:
    """Set to -inf the illegal logits"""
    masked = logits.copy()
    for token_id in range(len(masked)):
        if token_id not in valid_ids:
            masked[token_id] = float("-inf")
    return masked


def select_best_token(logits: list[float], valid_ids: set[int]) -> int:
    """Select token with the highest score"""
    if not valid_ids:
        raise ValueError("No valid token available")
    return max(valid_ids, key=lambda token_id: logits[token_id])


# only keep tokens that continue a target
def get_valid_token_ids(vocab: dict[str, int], segment: str,
                        targets: list[str]) -> set[int]:
    """Constrained decoding to only allow tokens with corresponding prefix"""
    valid_ids: set[int] = set()
    for token, token_id in vocab.items():
        candidate = segment + token
        for target in targets:
            if target.startswith(candidate):
                valid_ids.add(token_id)
                break
    return valid_ids


def value_written(segment: str, param_type: str) -> bool:
    """The parameter value has been written ? it has to end with ',' or '}'"""
    if segment == "" or segment[-1] not in ",}":
        return False
    if param_type == "string":
        return segment.count('"') == 2
    return True


def get_parameter_valid_ids(vocab: dict[str, int], segment: str, name: str,
                            key_written: bool, is_last: bool, param_type: str,
                            safe_ids: set[int]) -> set[int]:
    """Which tokens are valid for the current parameters ?"""
    if not key_written:
        return get_valid_token_ids(vocab, segment, [f'"{name}":'])
    if param_type == "string":
        return get_string_value_ids(vocab, segment, is_last, safe_ids)
    return get_valid_number_token_ids(vocab, segment, is_last)
