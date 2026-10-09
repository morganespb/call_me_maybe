MAX_STRING_LEN = 50


def number_next_chars(value: str, is_last: bool) -> set[str]:
    """Which tokens are allowed inside a number ?"""
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
    """Which tokens are allowed in our current number ?"""
    allowed = number_next_chars(value, is_last)
    valid_ids = set()
    for token, token_id in vocab.items():
        if token in allowed:
            valid_ids.add(token_id)
    return valid_ids


def get_string_safe_ids(token_texts: dict[int, str]) -> set[int]:
    """Which tokens are allowed inside a string ?"""
    safe_ids: set[int] = set()
    for token_id, text in token_texts.items():
        if text and '"' not in text and "\\" not in text \
                and text.isprintable():
            safe_ids.add(token_id)
    return safe_ids


def get_string_value_ids(vocab: dict[str, int], segment: str, is_last: bool,
                         safe_ids: set[int]) -> set[int]:
    """Which tokens are allowed in our current string ?"""
    quote_id = vocab['"']
    comma_id = vocab[',']
    brace_id = vocab['}']
    quote_count = segment.count('"')

    if quote_count == 0:
        return {quote_id}
    if quote_count == 1:
        end_token = '"}' if is_last else '",'
        if len(segment) > MAX_STRING_LEN:
            return {quote_id}
        else:
            valid_ids = set(safe_ids)
            valid_ids.add(quote_id)
        if end_token in vocab:
            valid_ids.add(vocab[end_token])
        return valid_ids
    if is_last:
        return {brace_id}
    return {comma_id}
