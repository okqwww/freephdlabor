import os
import yaml
import functools

# Custom parameter filtering function for model-specific requirements
# Note: NewAPI handles most parameter filtering automatically.
# This decorator provides additional safety for edge cases.
# Only NewAPI supported models: gpt-5, gpt-5, gpt-5-mini, gpt-5-nano
def filter_model_params(original_func):
    """Decorator to filter unsupported parameters for different models.

    NewAPI (newapi.tsingyuai.com/v1) handles model routing and parameter
    filtering automatically. This decorator provides additional safety
    for edge cases and maintains compatibility with the codebase.

    Only supports: gpt-5, gpt-5, gpt-5-mini, gpt-5-nano
    """
    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        model = kwargs.get('model', args[0] if args else '')

        # GPT-5 specific filtering
        if isinstance(model, str) and model.startswith('gpt-5'):
            # Remove unsupported GPT-5 parameters
            unsupported_params = {
                'stop', 'temperature', 'top_p', 'presence_penalty',
                'frequency_penalty', 'logprobs', 'top_logprobs',
                'logit_bias', 'max_tokens'
            }

            # Filter out unsupported parameters
            filtered_kwargs = {k: v for k, v in kwargs.items()
                             if k not in unsupported_params}

            # Replace max_tokens with max_completion_tokens if present
            if 'max_tokens' in kwargs:
                filtered_kwargs['max_completion_tokens'] = kwargs['max_tokens']

            return original_func(*args, **filtered_kwargs)

        else:
            # gpt-5 and other models use original parameters
            return original_func(*args, **kwargs)
    return wrapper

def load_llm_config():
    """Load LLM configuration from .llm_config.yaml if it exists."""
    config_path = ".llm_config.yaml"
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"📖 Loaded LLM config from {config_path}")
            return config
        except yaml.YAMLError as e:
            print(f"⚠️ Error loading {config_path}: {e}")
            return None
    else:
        print(f"ℹ️ No {config_path} found, using CLI arguments")
        return None
