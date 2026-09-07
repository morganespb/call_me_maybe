# prompt -> encode -> [IDs] -> Qwen gives logit -> append token ID-> repeat -> decode all IDS.
from llm_sdk import Small_LLM_Model
from src.decoder import DecoderState, get_valid_token_ids, next_state, select_best_token, get_targets
from .models import FunctionDefinition

def generate_tokens(model: Small_LLM_Model, prompt: str, vocab: dict[str, int],
                    functions: list[FunctionDefinition],
                    max_tokens: int = 50) -> str:
    input_ids = model.encode(prompt)
    generated_ids = input_ids[0].tolist()
    function_names = [function.name for function in functions]

    state = DecoderState.START
    current = ""
    segment = ""
    selected_function = None

    for _ in range(max_tokens):
        targets = get_targets(state, function_names)

        logits = model.get_logits_from_input_ids(generated_ids)

        valid_ids = get_valid_token_ids(vocab, segment, targets)
        next_token_id = select_best_token(logits, valid_ids) 
        
        generated_ids.append(next_token_id)

        token_text = model.decode([next_token_id])
        current += token_text
        segment += token_text

        if segment in targets:
            if state == DecoderState.FUNCTION_NAME:
                selected_name = segment.strip('"')

                for function in functions:
                    if function.name == selected_name:
                        selected_function = function
                        parameters_names = list(function.parameters.keys())
                        break

            segment = ""
            state = next_state(state)
            if state == DecoderState.PARAMETERS:
                break       
    return current
