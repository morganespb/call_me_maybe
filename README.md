

models.py -> defines the shape of the data using pydantic
    what is a valid prompt ?
    what is a valid function ?
    what should a result looks like ?

    class Prompt: represents one JSON input.
    class ParameterDefinition: represents parameter type (int, number, string)
    class FunctionDefinition: represents whole availaible function

parsing.py -> read JSON input files and converts them into validated pydantic objects
    load JSON = file path -> open_file -> json.load() -> python object
    load functions = function_definitions.json -> load_jason() -> list of dictionaries -> pydantic validation -> list[FunctionDefinition]



