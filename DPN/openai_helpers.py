import os
from typing import Any, TypeVar
from typing import Dict, Optional, Type
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers.pydantic import TBaseModel
from langchain_openai.chat_models import ChatOpenAI
T = TypeVar('T', bound=BaseModel)

def call_gpt_with_prompt_model(
    api_key: str | None,
    prompt: str,
    pydantic_object: Optional[Type[TBaseModel]],
    model: str = "gpt-4o",
) -> Dict:
    """
    Generic function to ask for GPT's inference. Return value will be in pydantic_object
    :param api_key:
    :param prompt:
    :param pydantic_object:
    :return:
    """
    if api_key is None:
        api_key = os.environ("OPEN_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY is not set")
    
    chat = ChatOpenAI(model=model, api_key=api_key, temperature=0.2)
    parser = JsonOutputParser(pydantic_object=pydantic_object)
    format_instructions = parser.get_format_instructions()
    _prompt = f"Answer the user query.\n{format_instructions}\n{prompt}\n"
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": _prompt},
            ]
        )
    ]

    text_result = chat.invoke(messages)
    return parser.invoke(text_result)


def call_llm(
    prompt_template: str,
    model: type[T],
    params: dict[str, Any] | None = None
) -> T:
    if params is None:
        params = {}
    RETRY_COUNT = 0
    error_message = ""
    while RETRY_COUNT < 3:
        try:
            res_dict = call_gpt_with_prompt_model(
                api_key=os.getenv("OPENAI_API_KEY"),
                prompt=prompt_template.format(**params),
                pydantic_object=model
            )
            return model.model_validate(res_dict)
        except Exception as e:
            error_message = str(e)
            RETRY_COUNT += 1
            print(f"Error: {e}")
            print(f"Retrying... ({RETRY_COUNT}/3)")

    raise Exception(f"Failed to call LLM after {RETRY_COUNT} attempts because of {error_message}")


def call_llm_text_response(
        prompt_template: str, params: dict[str, Any] | None = None) -> str:
    if params is None:
        params = {}

    class TextResponse(BaseModel):
        text: str

    res_dict = call_gpt_with_prompt_model(
        api_key=os.getenv("OPENAI_API_KEY"),
        prompt=prompt_template.format(**params),
        pydantic_object=TextResponse
    )

    return res_dict['text']