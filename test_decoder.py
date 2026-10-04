from src.decoder import get_valid_token_ids, DecoderState, next_state, get_targets, number_next_chars

# right states order ?
print(next_state(DecoderState.START))
print(next_state(DecoderState.FUNCTION_NAME))
print(next_state(DecoderState.PARAMETERS_START))

# right targets ?
print(get_targets(DecoderState.START, []))
print(get_targets(DecoderState.FUNCTION_NAME, ["fn_add", "fn_greet"]))
print(get_targets(DecoderState.PARAMETERS_START, []))

# legal tokens ?
vocab = {'"fn': 1, "_add": 2, "x": 3}
targets = ['"fn_add"', '"fn_greet"']
print(get_valid_token_ids(vocab, "", targets))
print(get_valid_token_ids(vocab, '"fn', targets))

# sorting numbers ?

print(sorted(number_next_chars("", False)))
print(sorted(number_next_chars("12", True)))
print(sorted(number_next_chars("12.", False)))

# path = model.get_path_to_vocab_file()
# vocab = load_vocabulary(path)

# print(vocab.get("m"))
# print(vocab.get("me"))

# print('"name"'.startswith('"name"'))
# print('"name"'.startswith('"name"'))

# targets = ['"name"']
# current = '"na'

# valid = get_valid_token_ids(vocab, current, targets)
# for token, token_id in vocab.items():
#    if token_id in valid:
#        print(repr(token), token_id)
