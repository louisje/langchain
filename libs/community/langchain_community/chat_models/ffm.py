"""Wrapper LLM conversation APIs."""
from typing import Any, Dict, List, Mapping, Optional, Tuple, AsyncIterator, Union
from pydantic import BaseModel, Field
from langchain_core.language_models.base import BaseLanguageModel

from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.messages import AIMessageChunk
from langchain_core.language_models.chat_models import (
    SimpleChatModel,
    agenerate_from_stream,
)

from langchain.schema import (
    BaseMessage,
    ChatMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    FunctionMessage,
)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)

import requests
import json
import re


class BaseFormosaFoundationModel(BaseLanguageModel):
    base_url: str = "http://localhost:12345"
    """Base url the model is hosted under."""

    model: str = "ffm-llama2-70b-chat"
    """Model name to use."""

    temperature: Optional[float]
    """The temperature of the model. Increasing the temperature will
    make the model answer more creatively."""

    stop: Optional[List[str]]
    """Sets the stop tokens to use."""

    top_k: int = 50
    """Reduces the probability of generating nonsense. A higher value (e.g. 100)
    will give more diverse answers, while a lower value (e.g. 10)
    will be more conservative. (Default: 50)"""

    top_p: float = 1
    """Works together with top-k. A higher value (e.g., 0.95) will lead
    to more diverse text, while a lower value (e.g., 0.5) will
    generate more focused and conservative text. (Default: 1)"""

    max_new_tokens: int = 350
    """The maximum number of tokens to generate in the completion.
    -1 returns as many tokens as possible given the prompt and
    the models maximal context size."""

    frequence_penalty: float = 1
    """Penalizes repeated tokens according to frequency."""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""

    ffm_api_key: Optional[str] = None

    streaming: bool = False

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling FFM API."""
        normal_params = {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "frequence_penalty": self.frequence_penalty,
            "top_k": self.top_k,
        }
        return {**normal_params, **self.model_kwargs}

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):

        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            stop = self.stop
        elif stop is None:
            stop = []
        params = {**self._default_params, "stop": stop, **kwargs}
        parameter_payload = {
            "parameters": params,
            "messages": [self._convert_message_to_dict(m) for m in messages],
            "model": self.model,
        }

        # HTTP headers for authorization
        headers = {
            "X-API-KEY": self.ffm_api_key,
            "Content-Type": "application/json",
        }
        endpoint_url = f"{self.base_url}/api/models/conversation"
        # send request
        try:
            response = requests.post(
                url=endpoint_url,
                headers=headers,
                data=json.dumps(parameter_payload, ensure_ascii=False).encode("utf8"),
            )
            if response.status_code != 200:
                raise ValueError(
                    f"FormosaFoundationModel endpoint_url: {endpoint_url}\n"
                    f"error raised with status code {response.status_code}\n"
                    f"Details: {response.text}\n"
                )
            response.encoding = "utf-8"
            generated_text = response.json()

        except requests.exceptions.RequestException as e:  # This is the correct syntax
            raise ValueError(
                f"FormosaFoundationModel error raised by inference endpoint: {e}\n"
            )

        if generated_text.get("detail", None) is not None:
            detail = generated_text["detail"]
            raise ValueError(
                f"FormosaFoundationModel endpoint_url: {endpoint_url}\n"
                f"error raised by inference API: {detail}\n"
            )

        if generated_text.get("generated_text", None) is None:
            raise ValueError(
                f"FormosaFoundationModel endpoint_url: {endpoint_url}\n"
                f"Response format error: {generated_text}\n"
            )

        return generated_text

    async def _acall(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            stop = self.stop
        elif stop is None:
            stop = []
        params = {**self._default_params, "stop": stop, **kwargs}
        parameter_payload = {
            "parameters": params,
            "messages": [self._convert_message_to_dict(m) for m in messages],
            "model": self.model,
            "stream": True,
        }

        # HTTP headers for authorization
        headers = {
            "X-API-KEY": self.ffm_api_key,
            "Content-Type": "application/json",
        }
        endpoint_url = f"{self.base_url}/api/models/conversation"
        # send request
        try:
            response = requests.post(
                url=endpoint_url,
                headers=headers,
                data=json.dumps(parameter_payload, ensure_ascii=False).encode("utf8"),
                stream=True,
            )
            if response.status_code != 200:
                raise ValueError(
                    f"FormosaFoundationModel endpoint_url: {endpoint_url}\n"
                    f"error raised with status code {response.status_code}\n"
                    f"Details: {response.text}\n"
                )
            response.encoding = "utf-8"

            for line in response.iter_lines():
                if len(line) == 0:
                    continue
                chunk: str = line.lstrip(b"data: ").decode("utf-8")
                if chunk == "[DONE]":
                    break
                if chunk == "event: ping" or not chunk:
                    continue
                if re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{6}", chunk):
                    continue
                data: dict[str, str] = json.loads(chunk)
                if (
                    "generated_text" not in data
                    or data["generated_text"] is None
                    or len(data["generated_text"]) == 0
                ):
                    continue
                yield data

        except requests.exceptions.RequestException as e:  # This is the correct syntax
            raise ValueError(
                f"FormosaFoundationModel error raised by inference endpoint: {e}\n"
            )

    def _convert_message_to_dict(self, message: BaseMessage) -> dict:
        if isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "content": message.content}
        elif isinstance(message, HumanMessage):
            message_dict = {"role": "human", "content": message.content}
        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}
        elif isinstance(message, SystemMessage):
            message_dict = {"role": "system", "content": message.content}
        elif isinstance(message, FunctionMessage):
            message_dict = {
                "role": "human",
                "content": f"呼叫工具`{message.name}`之後我們得到以下回覆：\n\n{message.content}",
            }
        else:
            raise ValueError(f"Got unknown type {message}")
        return message_dict



class ChatFFM(BaseFormosaFoundationModel, SimpleChatModel):
    """`FormosaFoundation` Chat large language models API.

    The environment variable ``OPENAI_API_KEY`` set with your API key.

    Example:
        .. code-block:: python
            ffm = ChatFFM(model="meta-llama2-70b-chat")
    """

    @property
    def _llm_type(self) -> str:
        return "FFM-chat"

    @property
    def lc_serializable(self) -> bool:
        return True

    def _create_conversation_messages(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params: Dict[str, Any] = {**self._default_params}

        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop

        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(self, response: Union[dict, BaseModel]) -> ChatResult:
        if not isinstance(response, dict):
            response = response.dict()
        generation = ChatGeneration(
            message=AIMessage(content=response.get("generated_text", "")),
            generation_info=dict(finish_reason=response.get("finish_reason")),
        )
        llm_output = {
            "token_usage": response.get("generated_tokens"),
            "model_name": self.model,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }
        return ChatResult(generations=[generation], llm_output=llm_output)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        print(message_dicts)  # Qoo
        async for chunk in self._acall(messages=messages, stop=stop, **params):
            if not isinstance(chunk, dict):
                chunk = chunk.dict()
            ai_message_chunk = AIMessageChunk(content=chunk["generated_text"])
            finish_reason = chunk.get("finish_reason")
            generation_info = (
                dict(finish_reason=finish_reason) if finish_reason is not None else None
            )
            chunk = ChatGenerationChunk(
                message=ai_message_chunk, generation_info=generation_info
            )
            yield chunk
            if run_manager:
                await run_manager.on_llm_new_token(token=chunk.text, chunk=chunk)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        print(json.dumps(message_dicts))  # Qoo
        params = {**params, **kwargs}
        response = self._call(messages=messages, stop=stop, **params)
        print(json.dumps(response))  # Qoo
        if type(response) is str:  # response is not the format of dictionary
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])
        return self._create_chat_result(response)

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        return message_dicts, params
