import importlib_metadata
from tokenizers import processors
from transformers import PreTrainedTokenizerFast
from vllm import LLM, SamplingParams
from vllm.logger import init_logger

from serving.base_serving import BaseServing

logger = init_logger(__name__)


def force_support_for_add_bos_token(tokenizer: PreTrainedTokenizerFast) -> None:
    """
    Hack to incorporate:

    https://github.com/huggingface/transformers/pull/31316
    """

    text = "a"
    tokens_default: list[int] = tokenizer(text)["input_ids"]

    # We need to initialize these correctly, not None. The reason is that if we update
    # set add_eos/bos_token later, and then reset it back to None, we'll always have
    # False-y values instead of the original behavior.
    tokenizer._add_eos_token = tokens_default[-1] == getattr(
        tokenizer, "eos_token_id", None
    )
    tokenizer._add_bos_token = tokens_default[0] == getattr(
        tokenizer, "bos_token_id", None
    )

    class _PreTrainedTokenizerFastPatched(type(tokenizer)):
        @property
        def add_eos_token(self):
            return self._add_eos_token

        @property
        def add_bos_token(self):
            return self._add_bos_token

        @add_eos_token.setter
        def add_eos_token(self, value: bool):
            self._add_eos_token = value
            self.update_post_processor()

        @add_bos_token.setter
        def add_bos_token(self, value: bool):
            self._add_bos_token = value
            self.update_post_processor()

        def update_post_processor(self):
            """
            Overwrites the underlying post processor with the current `bos_token` and
            `eos_token`.
            """
            if not isinstance(
                self._tokenizer.post_processor, processors.TemplateProcessing
            ) and not isinstance(self._tokenizer.post_processor, processors.Sequence):
                return

            bos = self.bos_token
            bos_token_id = self.bos_token_id
            if bos is None and self.add_bos_token:
                raise ValueError("add_bos_token = True but bos_token = None")

            eos = self.eos_token
            eos_token_id = self.eos_token_id
            if eos is None and self.add_eos_token:
                raise ValueError("add_eos_token = True but eos_token = None")

            single = (
                f"{(bos + ':0 ') if self.add_bos_token else ''}"
                "$A:0"
                f"{(' ' + eos + ':0') if self.add_eos_token else ''}"
            )
            pair = (
                f"{single}{(' ' + bos + ':1') if self.add_bos_token else ''} "
                "$B:1"
                f"{(' ' + eos + ':1') if self.add_eos_token else ''}"
            )

            special_tokens = []
            if self.add_bos_token:
                special_tokens.append((bos, bos_token_id))
            if self.add_eos_token:
                special_tokens.append((eos, eos_token_id))
            self._tokenizer.post_processor = processors.TemplateProcessing(
                single=single, pair=pair, special_tokens=special_tokens
            )

    # https://stackoverflow.com/questions/31590152/monkey-patching-a-property
    tokenizer.__class__ = _PreTrainedTokenizerFastPatched


class VLLMServing(BaseServing):
    def __init__(
        self,
        model: str,
        is_base_model: bool = False,
        dtype="bfloat16",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        seed=1234,
        add_bos_token=False,
        **kwargs,
    ):
        if is_base_model:
            with open("chat_templates/base_model.jinja") as f:
                chat_template = f.read()
            self.chat_template = chat_template
        else:
            self.chat_template = None

        self.llm = LLM(
            model=model,
            dtype=dtype,
            enable_prefix_caching=enable_prefix_caching,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            **kwargs,
        )

        # hack to remove double bos token
        if add_bos_token == False:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model, add_bos_token=False)
            if "llama" in model:
                force_support_for_add_bos_token(tokenizer)
                tokenizer.add_bos_token = False

            self.llm.set_tokenizer(tokenizer)

    def get_run_env(self):
        return {
            "transformers_version": importlib_metadata.version("transformers"),
            "vllm_version": importlib_metadata.version("vllm"),
            # "vllm_config": str(self.llm.llm_engine.get_model_config()),
        }

    def generate(self, messages: list, logprobs: bool = False, **generation_kwargs):
        response = self.llm.chat(
            messages=messages,
            sampling_params=SamplingParams(**generation_kwargs),
            chat_template=self.chat_template,
            add_generation_prompt=True,
        )
        return response

    def batch_generate(
        self,
        batch_messages: list[list],
        logprobs: bool = False,
        **generation_kwargs,
    ):
        responses = self.llm.chat(
            messages=batch_messages,
            sampling_params=SamplingParams(**generation_kwargs),
            chat_template=self.chat_template,
            add_generation_prompt=True,
        )
        return responses

    def parse_outputs(self, generated_outputs, **kwargs):
        responses = []
        errors = []
        tokenized_prompts = []

        for output in generated_outputs:
            responses.append(output.outputs[0].text)
            if output.outputs[0].text == "":
                # Log empty string as an EmptyGenerationError
                errors.append("EmptyGenerationError")
            else:
                errors.append(None)
            tokenized_prompts.append(output.prompt_token_ids)

        return responses, errors, tokenized_prompts
