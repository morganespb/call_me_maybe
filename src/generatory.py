# prompt -> encode -> [IDs] -> Qwen gives logit -> append token ID->
# repeat -> decode all IDS.
from llm_sdk import Small_LLM_Model
from src.decoder import (DecoderState, get_valid_number_token_ids,
                         get_valid_token_ids, next_state, select_best_token,
                         get_targets, value_written)
from .models import FunctionDefinition


def build_prompt(prompt: str, functions: list[FunctionDefinition]) -> str:
    """ Build the texts QWEN reads: the functions and the request """
    text = "Choose the function that answers the request. \n"
    for function in functions:
        text += function.name + ": " + function.description + "\n"
    text += "Request: " + prompt + "\n"
    text += "Answer as JSON:\n"
    return text


def generate_tokens(model: Small_LLM_Model, prompt: str, vocab: dict[str, int],
                    functions: list[FunctionDefinition],
                    max_tokens: int = 50) -> str:
    """Generate JSON function call for one prompt"""
    input_ids = model.encode(build_prompt(prompt, functions))
    generated_ids = input_ids[0].tolist()
    function_names = [function.name for function in functions]

    state = DecoderState.START
    current = ""  # full JSON text
    segment = ""  # text at this stage
    parameter_names: list[str] = []
    parameter_index = 0
    key_written = False

    for _ in range(max_tokens):
        targets = get_targets(state, function_names)
        logits = model.get_logits_from_input_ids(generated_ids)

        if state == DecoderState.PARAMETERS:
            name = parameter_names[parameter_index]
            is_last = parameter_index == len(parameter_names) - 1
            if not key_written:
                valid_ids = get_valid_token_ids(vocab, segment, [f'"{name}":'])
            else:
                valid_ids = get_valid_number_token_ids(vocab, segment, is_last)
        else:
            valid_ids = get_valid_token_ids(vocab, segment, targets)

        next_token_id = select_best_token(logits, valid_ids)
        generated_ids.append(next_token_id)
        token_text = model.decode([next_token_id])
        current += token_text
        segment += token_text

        if state == DecoderState.PARAMETERS:
            if not key_written and segment == f'"{name}":':
                key_written = True
                segment = ""
            elif key_written and value_written(segment):
                if is_last:
                    current += "}"
                    break
                parameter_index += 1
                key_written = False
                segment = ""

        elif segment in targets:  # is the stage finished ?
            if state == DecoderState.FUNCTION_NAME:
                selected_name = segment.strip('"')
                for function in functions:
                    if function.name == selected_name:
                        parameter_names = list(function.parameters.keys())
                        break

            segment = ""
            state = next_state(state)
            if state == DecoderState.PARAMETERS and not parameter_names:
                current += "}}"
                break
    return current
