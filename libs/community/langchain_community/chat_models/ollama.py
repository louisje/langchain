import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union, cast
from langchain_core.prompts.prompt import PromptTemplate

from langchain_experimental.llms.ollama_functions import DEFAULT_RESPONSE_FUNCTION, OllamaFunctions

from langchain_core._api import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel, LangSmithParams
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    FunctionMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from langchain_community.llms.ollama import Ollama, OllamaEndpointNotFoundError


@deprecated("0.0.3", alternative="_chat_stream_response_to_chat_generation_chunk")
def _stream_response_to_chat_generation_chunk(
    stream_response: str,
) -> ChatGenerationChunk:
    """Convert a stream response to a generation chunk."""
    parsed_response = json.loads(stream_response)
    generation_info = parsed_response if parsed_response.get("done") is True else None
    return ChatGenerationChunk(
        message=AIMessageChunk(content=parsed_response.get("response", "")),
        generation_info=generation_info,
    )


def _chat_stream_response_to_chat_generation_chunk(
    stream_response: str,
) -> ChatGenerationChunk:
    """Convert a stream response to a generation chunk."""
    parsed_response = json.loads(stream_response)
    generation_info = parsed_response if parsed_response.get("done") is True else None
    return ChatGenerationChunk(
        message=AIMessageChunk(
            content=parsed_response.get("message", {}).get("content", "")
        ),
        generation_info=generation_info,
    )


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    alternative_import="langchain_ollama.ChatOllama",
)
class ChatOllama(Ollama, OllamaFunctions):
    """Ollama locally runs large language models.

    To use, follow the instructions at https://ollama.ai/.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatOllama
            ollama = ChatOllama(model="llama2")
    """

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "ollama-chat"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return False

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="ollama",
            ls_model_name=self.model,
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("num_predict", self.num_predict):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None) or self.stop:
            ls_params["ls_stop"] = ls_stop
        return ls_params

    @deprecated("0.0.3", alternative="_convert_messages_to_ollama_messages")
    def _format_message_as_text(self, message: BaseMessage) -> str:
        if isinstance(message, ChatMessage):
            message_text = f"\n\n{message.role.capitalize()}: {message.content}"
        elif isinstance(message, HumanMessage):
            if isinstance(message.content, List):
                first_content = cast(List[Dict], message.content)[0]
                content_type = first_content.get("type")
                if content_type == "text":
                    message_text = f"[INST] {first_content['text']} [/INST]"
                elif content_type == "image_url":
                    message_text = first_content["image_url"]["url"]
            else:
                message_text = f"[INST] {message.content} [/INST]"
        elif isinstance(message, AIMessage):
            message_text = f"{message.content}"
        elif isinstance(message, SystemMessage):
            message_text = f"<<SYS>> {message.content} <</SYS>>"
        else:
            raise ValueError(f"Got unknown type {message}")
        return message_text

    def _format_messages_as_text(self, messages: List[BaseMessage]) -> str:
        return "\n".join(
            [self._format_message_as_text(message) for message in messages]
        )

    def _convert_messages_to_ollama_messages(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Union[str, List[str]]]]:
        ollama_messages: List = []
        for message in messages:
            role = ""
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, FunctionMessage):
                role = "user"
            else:
                raise ValueError("Received unsupported message type for Ollama.")

            content = ""
            images = []
            if isinstance(message, FunctionMessage):
                tool_output = {
                    "tool_name": message.name,
                    "tool_output": message.content,
                }
                content = json.dumps(tool_output, indent=4)
                print(content) # DEBUG
            elif isinstance(message, AIMessage) and not message.content and "function_call" in message.additional_kwargs:
                function_call = {
                    "name": message.additional_kwargs["function_call"]["name"],
                    "arguments": message.additional_kwargs["function_call"]["arguments"],
                }
                content = json.dumps(function_call, indent=4)
            elif isinstance(message.content, str):
                content = message.content
            else:
                for content_part in cast(List[Dict], message.content):
                    if content_part.get("type") == "text":
                        content += f"\n{content_part['text']}"
                    elif content_part.get("type") == "image_url":
                        image_url = None
                        temp_image_url = content_part.get("image_url")
                        if isinstance(temp_image_url, str):
                            image_url = content_part["image_url"]
                        elif (
                            isinstance(temp_image_url, dict) and "url" in temp_image_url
                        ):
                            image_url = temp_image_url["url"]
                        else:
                            raise ValueError(
                                "Only string image_url or dict with string 'url' "
                                "inside content parts are supported."
                            )

                        image_url_components = image_url.split(",")
                        # Support data:image/jpeg;base64,<image> format
                        # and base64 strings
                        if len(image_url_components) > 1:
                            images.append(image_url_components[1])
                        else:
                            images.append(image_url_components[0])

                    else:
                        raise ValueError(
                            "Unsupported message content type. "
                            "Must either have type 'text' or type 'image_url' "
                            "with a string 'image_url' field."
                        )

            ollama_messages.append(
                {
                    "role": role,
                    "content": content,
                    "images": images,
                }
            )
        print(ollama_messages) # DEBUG
        return ollama_messages


    def _create_chat_stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": self._convert_messages_to_ollama_messages(messages),
        }
        yield from self._create_stream(
            payload=payload, stop=stop, api_url=f"{self.base_url}/api/chat", **kwargs
        )

    async def _acreate_chat_stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": self._convert_messages_to_ollama_messages(messages),
        }
        async for stream_resp in self._acreate_stream(
            payload=payload, stop=stop, api_url=f"{self.base_url}/api/chat", **kwargs
        ):
            yield stream_resp

    def _chat_stream_with_aggregation(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> ChatGenerationChunk:
        final_chunk: Optional[ChatGenerationChunk] = None
        for stream_resp in self._create_chat_stream(messages, stop, **kwargs):
            if stream_resp:
                chunk = _chat_stream_response_to_chat_generation_chunk(stream_resp)
                if final_chunk is None:
                    final_chunk = chunk
                else:
                    final_chunk += chunk
                if run_manager:
                    run_manager.on_llm_new_token(
                        token=chunk.text,
                        chunk=chunk,
                        verbose=verbose,
                    )
        if final_chunk is None:
            raise ValueError("No data received from Ollama stream.")

        return final_chunk

    async def _achat_stream_with_aggregation(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> ChatGenerationChunk:
        final_chunk: Optional[ChatGenerationChunk] = None
        async for stream_resp in self._acreate_chat_stream(messages, stop, **kwargs):
            if stream_resp:
                chunk = _chat_stream_response_to_chat_generation_chunk(stream_resp)
                if final_chunk is None:
                    final_chunk = chunk
                else:
                    final_chunk += chunk
                if run_manager:
                    await run_manager.on_llm_new_token(
                        token=chunk.text,
                        chunk=chunk,
                        verbose=verbose,
                    )
        if final_chunk is None:
            raise ValueError("No data received from Ollama stream.")

        return final_chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call out to Ollama's generate endpoint.

        Args:
            messages: The list of base messages to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            Chat generations from the model

        Example:
            .. code-block:: python

                response = ollama([
                    HumanMessage(content="Tell me about the history of AI")
                ])
        """
        functions = kwargs.get("functions", [])
        if "function_call" in kwargs:
            functions = [
                fn for fn in functions if fn["name"] == kwargs["function_call"]["name"]
            ]
            if not functions:
                raise ValueError(
                    'If "function_call" is specified, you must also pass a matching function in "functions".'
                )
            del kwargs["function_call"]
        if "functions" in kwargs:
            del kwargs["functions"]
            if isinstance(messages[-1], HumanMessage) and functions:
                last_message = messages.pop()
                tool_prompt = PromptTemplate.from_template(self.tool_system_prompt_template)
                prompt = tool_prompt.format(tools=json.dumps(functions, indent=4))
                messages.append(HumanMessage(content=prompt+"\n\nMy question is:\n"+str(last_message.content)+"\n\nLet's think step by step."))
        chat_generation = self._chat_stream_with_aggregation(
            messages,
            stop=stop,
            run_manager=run_manager,
            verbose=self.verbose,
            **kwargs,
        )

        chat_generation_content = chat_generation.text
        if not isinstance(chat_generation_content, str):
            raise ValueError("OllamaFunctions does not support non-string output.")
        try:
            parsed_chat_result = json.loads(chat_generation_content)
            called_tool_name = parsed_chat_result["tool"]
            called_tool_arguments = parsed_chat_result["tool_input"]
            called_tool = next(
                (fn for fn in functions if fn["name"] == called_tool_name), None
            )
            if called_tool is None:
                raise ValueError(
                    f"Failed to parse a function call from {self.model} output: {chat_generation_content}"
                )
            if called_tool["name"] == DEFAULT_RESPONSE_FUNCTION["name"]:
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=called_tool_arguments["response"]))])
            response_message_with_functions = AIMessage(
                content="",
                additional_kwargs = {
                    "function_call": {
                        "name": called_tool_name,
                        "arguments": json.dumps(called_tool_arguments)
                        if called_tool_arguments
                        else "",
                    },
                },
            )

            return ChatResult(generations=[ChatGeneration(message=response_message_with_functions)])
        except json.JSONDecodeError:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=chat_generation_content))])


    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call out to Ollama's generate endpoint.

        Args:
            messages: The list of base messages to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            Chat generations from the model

        Example:
            .. code-block:: python

                response = ollama([
                    HumanMessage(content="Tell me about the history of AI")
                ])
        """
        final_chunk = await self._achat_stream_with_aggregation(
            messages,
            stop=stop,
            run_manager=run_manager,
            verbose=self.verbose,
            **kwargs,
        )
        chat_generation = ChatGeneration(
            message=AIMessage(content=final_chunk.text),
            generation_info=final_chunk.generation_info,
        )
        return ChatResult(generations=[chat_generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        try:
            for stream_resp in self._create_chat_stream(messages, stop, **kwargs):
                if stream_resp:
                    chunk = _chat_stream_response_to_chat_generation_chunk(stream_resp)
                    if run_manager:
                        run_manager.on_llm_new_token(
                            token=chunk.text,
                            chunk=chunk,
                            verbose=self.verbose,
                        )
                    yield chunk
        except OllamaEndpointNotFoundError:
            yield from self._legacy_stream(messages, stop, **kwargs)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if "tools" in kwargs:
            tools = kwargs["tools"]
            del kwargs["tools"]
            if isinstance(messages[-1], HumanMessage) and tools:
                last_message = messages.pop()
                prompt_template = PromptTemplate.from_template(self.tool_system_prompt_template)
                prompt = prompt_template.format(tools=json.dumps(tools, indent=4))
                messages.append(HumanMessage(content=prompt+"\n\nMy question is:\n\n"+str(last_message.content)+"\n\nLet's think step by step."))

        async for stream_resp in self._acreate_chat_stream(messages, stop, **kwargs):
            if stream_resp:
                chunk = _chat_stream_response_to_chat_generation_chunk(stream_resp)
                if run_manager:
                    await run_manager.on_llm_new_token(
                        token=chunk.text,
                        chunk=chunk,
                        verbose=self.verbose,
                    )
                yield chunk

    @deprecated("0.0.3", alternative="_stream")
    def _legacy_stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        prompt = self._format_messages_as_text(messages)
        for stream_resp in self._create_generate_stream(prompt, stop, **kwargs):
            if stream_resp:
                chunk = _stream_response_to_chat_generation_chunk(stream_resp)
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        chunk=chunk,
                        verbose=self.verbose,
                    )
                yield chunk
