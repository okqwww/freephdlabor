import json
import os
import re
from typing import Any
from ai_scientist.utils.token_tracker import track_token_usage

import backoff
import openai

MAX_NUM_TOKENS = 4096

# Available LLMs via NewAPI (newapi.tsingyuai.com/v1)
# All models use OpenAI SDK with NewAPI endpoint
AVAILABLE_LLMS = [
    # NewAPI supported models (primary)
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5",
    "gpt-4o",
    "gpt-4o-mini",
    # Claude models via NewAPI
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    "claude-sonnet-4-5",
    # Legacy OpenAI models
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "o1-preview-2024-09-12",
    "o1-mini-2024-09-12",
    "o1-2024-12-17",
    "o3-mini-2025-01-31",
    # Gemini models
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]


# Get N responses from a single message, used for ensembling.
@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
    ),
)
@track_token_usage
def get_batch_responses_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
    n_responses=1,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    """Get multiple responses from LLM via NewAPI for ensembling."""
    msg = prompt
    if msg_history is None:
        msg_history = []

    new_msg_history = msg_history + [{"role": "user", "content": msg}]

    # Use OpenAI SDK format for all models via NewAPI
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            *new_msg_history,
        ],
        temperature=temperature,
        max_tokens=MAX_NUM_TOKENS,
        n=n_responses,
    )
    content = [r.message.content for r in response.choices]
    new_msg_history = [
        new_msg_history + [{"role": "assistant", "content": c}] for c in content
    ]

    if print_debug:
        # Just print the first one.
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


@track_token_usage
def make_llm_call(client, model, temperature, system_message, prompt):
    """Make LLM call via NewAPI using OpenAI SDK format."""
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            *prompt,
        ],
        temperature=temperature,
        max_tokens=MAX_NUM_TOKENS,
        n=1,
    )


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
    ),
)
def get_response_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
) -> tuple[str, list[dict[str, Any]]]:
    """Get response from LLM via NewAPI.

    All models use OpenAI SDK with NewAPI endpoint.
    Model-specific parameter handling is done by NewAPI.
    """
    msg = prompt
    if msg_history is None:
        msg_history = []

    new_msg_history = msg_history + [{"role": "user", "content": msg}]

    # Use OpenAI SDK format for all models via NewAPI
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            *new_msg_history,
        ],
        temperature=temperature,
        max_tokens=MAX_NUM_TOKENS,
        n=1,
    )
    content = response.choices[0].message.content
    new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def extract_json_between_markers(llm_output: str) -> dict | None:
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found


def create_client(model) -> tuple[Any, str]:
    """Create an OpenAI client configured for NewAPI.

    All models are accessed via NewAPI (newapi.tsingyuai.com/v1) using OpenAI SDK.
    Model fallback and retry are handled by NewAPI, not in code.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_base = os.environ.get("OPENAI_BASE_URL", "https://newapi.tsingyuai.com/v1")

    # All models use NewAPI with OpenAI SDK
    print(f"Using NewAPI ({api_base}) with model {model}.")
    return openai.OpenAI(api_key=api_key, base_url=api_base), model
