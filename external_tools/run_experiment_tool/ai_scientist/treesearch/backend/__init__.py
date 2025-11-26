import os
from . import backend_openai
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query via NewAPI backend with a single system and user message.
    Supports function calling for some backends.

    Only NewAPI supported models: gpt-4o, gpt-5, gpt-5-mini, gpt-5-nano

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI format)
        model (str): string identifier for the model to use (gpt-4o, gpt-5, gpt-5-mini, gpt-5-nano)
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
    }

    # Handle GPT-5 models with reasoning_effort
    if model.startswith("gpt-5"):
        # GPT-5 models: Support system messages but not temperature
        # Check environment for reasoning_effort setting (from .llm_config.yaml)
        reasoning_effort = os.environ.get('RUN_EXPERIMENT_REASONING_EFFORT', 'high')
        model_kwargs["reasoning_effort"] = reasoning_effort
        model_kwargs.pop("temperature", None)  # GPT-5 doesn't support temperature
    else:
        # gpt-4o and other models
        model_kwargs["max_tokens"] = max_tokens

    # All models use OpenAI backend via NewAPI
    output, req_time, in_tok_count, out_tok_count, info = backend_openai.query(
        system_message=compile_prompt_to_md(system_message) if system_message else None,
        user_message=compile_prompt_to_md(user_message) if user_message else None,
        func_spec=func_spec,
        **model_kwargs,
    )

    return output
