import os
from dataclasses import dataclass
from openai import OpenAI
import logging


@dataclass
class ModelConfig:
    model: str = "deepseek-ai/DeepSeek-V3"
    max_tokens: int = 1000
    temperature: float = 0.1
    top_p: float = 0.1
    base_url: str = "https://router.huggingface.co/together"

    def __repr__(self):
        return (
            f"ModelConfig(\n"
            f"    model='{self.model}',\n"
            f"    max_tokens={self.max_tokens},\n"
            f"    temperature={self.temperature},\n"
            f"    top_p={self.top_p},\n"
            f"    base_url='{self.base_url}'\n"
            f")"
        )


def get_completion(
    prompt: str,
    config: ModelConfig = None
) -> str:
    if not config:
        config = ModelConfig()
    
    logging.debug(f"Using ModelConfig: {config}")

    api_token = os.getenv("HUGGING_FACE_API_TOKEN")
    if not api_token:
        raise ValueError(
            "HUGGING_FACE_API_TOKEN environment variable not found. "
            "Please set it in your .env file or environment variables."
        )

    client = OpenAI(
        base_url=config.base_url,
        api_key=api_token
    )

    messages = [{"role": "user", "content": prompt}]
    logging.debug(f"Sending prompt: {prompt}")
    
    completion = client.chat.completions.create(
        model=config.model,
        messages=messages,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p
    )

    logging.info(f"Completion successful with tokens {completion.usage.total_tokens}")
    return completion.choices[0].message.content


if __name__ == "__main__":
    response = get_completion("What is the capital of Malaysia?")
    print(response)
    logging.info(f"Response: {response}")
    
    custom_config = ModelConfig(
        model="different-model",
        temperature=0.7,
        max_tokens=100
    )
    response = get_completion("What is 2+2?", custom_config)
    print(response)
    logging.info(f"Response: {response}")
