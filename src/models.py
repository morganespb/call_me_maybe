from pydantic import BaseModel


class Prompt(BaseModel):
    prompt: str


class ParameterDefinition(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: dict[str, str]


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, object]
