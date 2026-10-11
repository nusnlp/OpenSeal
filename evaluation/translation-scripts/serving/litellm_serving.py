import asyncio
import logging
from urllib.parse import urlparse

import httpx
import importlib_metadata
import litellm
import requests
from litellm.exceptions import LITELLM_EXCEPTION_TYPES

from serving.base_serving import BaseServing


class LiteLLMServing(BaseServing):
    def __init__(
        self,
        model: str,
        base_url: str = None,
        api_key: str = None,
        is_base_model: bool = False,
        ssl_verify: bool = True,
        max_workers: int = 100,
        **kwargs,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.is_base_model = is_base_model
        self.max_workers = max_workers

        if self.is_base_model:
            logging.warning(
                "Base model selected. Please ensure that the chat template has been specified in the LiteLLM config."
            )

        if ssl_verify == False:
            litellm.client_session = httpx.Client(verify=False)

    def get_run_env(self):
        return {"litellm_version": importlib_metadata.version("litellm")}

    def generate(
        self,
        messages: list,
        logprobs: bool = False,
        num_retries: int = 10,
        **generation_kwargs,
    ):
        response = litellm.completion_with_retries(
            model=self.model,
            messages=messages,
            base_url=self.base_url,
            api_key=self.api_key,
            logprobs=logprobs,
            num_retries=num_retries,
            retry_strategy="exponential_backoff_retry",
            **generation_kwargs,
        )
        return response

    def tokenize(self, message):
        assert (
            self.base_url is not None
        ), "Base URL is required for to get tokenized prompts"

        parsed_url = urlparse(self.base_url)
        base_url = parsed_url.scheme + "://" + parsed_url.netloc

        model = "/".join(self.model.split("/")[1:])
        response = requests.post(
            base_url + "/tokenize",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "accept": "application/json",
            },
            json={
                "model": model,
                "messages": message,
                "add_special_tokens": False,
                "add_generation_prompt": True,
            },
        )
        return response.json()

    def batch_tokenize(self, messages: list):
        # TODO handle cases when encode does not work
        batch_response = [self.tokenize(message) for message in messages]

        return batch_response

    def batch_generate(
        self,
        batch_messages: list[list],
        logprobs: bool = False,
        **generation_kwargs,
    ):
        batch_response = litellm.batch_completion(
            model=self.model,
            messages=batch_messages,
            base_url=self.base_url,
            api_key=self.api_key,
            logprobs=logprobs,
            max_workers=self.max_workers,
            **generation_kwargs,
        )
        return batch_response

    async def agenerate(
        self, messages: list, logprobs: bool = False, **generation_kwargs
    ):
        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            base_url=self.base_url,
            api_key=self.api_key,
            logprobs=logprobs,
            **generation_kwargs,
        )
        return response

    def get_response(self, output):
        return output["choices"][0]["message"]["content"]

    def parse_outputs(
        self, generated_outputs, conversations=None, tokenize_prompts=False
    ):
        responses = []
        errors = []
        tokenized_prompts = []

        for output in generated_outputs:
            # Handle LiteLLM Error types
            if type(output) in LITELLM_EXCEPTION_TYPES:
                responses.append(None)
                errors.append(type(output).__name__)
            else:
                responses.append(self.get_response(output))
                errors.append(None)

        if tokenize_prompts:
            tokenized_prompts = self.batch_tokenize(conversations)

        return responses, errors, tokenized_prompts


if __name__ == "__main__":
    # start an OpenAI compatible server using vLLM with the following command
    # vllm serve google/gemma-2-9b-it --dtype bfloat16 --api-key token-abc123 --tensor-parallel-size 1 --enable-prefix-caching
    model = "openai/google/gemma-2-9b-it"
    base_url = "http://localhost:8000/v1"
    api_key = "token-abc123"

    litellmModel = LiteLLMServing(model, base_url, api_key, is_base_model=False)

    messages = [{"role": "user", "content": "ELI5: Why is the sky blue"}]
    # run generation
    response = litellmModel.generate(messages)
    print(response)

    # run batch generation
    response = litellmModel.batch_generate([messages for _ in range(5)])
    print(response)

    # run async generation
    response = asyncio.run(litellmModel.agenerate(messages))
    print(response)
