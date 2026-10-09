# prompt -> encode -> [IDs] -> Qwen gives logit -> append token ID->
# repeat -> decode all IDS.
from llm_sdk import Small_LLM_Model
from src.decoder import (DecoderState, get_valid_token_ids, next_state,
                         select_best_token, get_targets, value_written,
                         get_parameter_valid_ids)
from src.models import FunctionDefinition


def build_prompt(prompt: str, functions: list[FunctionDefinition]) -> str:
    """ Build the texts QWEN reads: the functions and the request """
    text = "Choose the function that answers the request. \n"
    for function in functions:
        text += function.name + ": " + function.description + "\n"

    text += "\nExample (not one of the functions):\n"
    text += "Request: Find the word 'quarante' in 'i love quarante deux'\n"
    text += 'Answer: {"name": "fn_example_find", "parameters":'
    text += '{"text": "i love quarante deux", "word": "quarante"}}\n'
    text += "Request: " + prompt + "\n"
    text += "Answer as JSON:\n"
    return text


def generate_tokens(model: Small_LLM_Model, prompt: str, vocab: dict[str, int],
                    functions: list[FunctionDefinition], safe_ids: set[int],
                    max_tokens: int = 200) -> str:
    """Generate JSON function call for one prompt.
    Get logits -> choose valid_ids
    -> pick a token -> update the state"""
    input_ids = model.encode(build_prompt(prompt, functions))
    generated_ids = input_ids[0].tolist()
    function_names = [function.name for function in functions]

    state = DecoderState.START
    current = ""  # full JSON text
    segment = ""  # text at this stage
    parameter_types: list[str] = []
    parameter_names: list[str] = []
    parameter_index = 0
    key_written = False

    for _ in range(max_tokens):
        targets = get_targets(state, function_names)
        logits = model.get_logits_from_input_ids(generated_ids)

        if state == DecoderState.PARAMETERS:
            name = parameter_names[parameter_index]
            is_last = parameter_index == len(parameter_names) - 1
            param_type = parameter_types[parameter_index]
            valid_ids = get_parameter_valid_ids(vocab, segment, name,
                                                key_written, is_last,
                                                param_type, safe_ids)
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
            elif key_written and value_written(segment, param_type):
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
                        for p in function.parameters.values():
                            parameter_types.append(p.type)
                        break

            segment = ""
            state = next_state(state)
            if state == DecoderState.PARAMETERS and not parameter_names:
                current += "}}"
                break
    return current
