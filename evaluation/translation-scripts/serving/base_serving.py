import time
from abc import ABC, abstractmethod


class BaseServing(ABC):
    @abstractmethod
    def generate(self, messages: list, logprobs: bool = False, **generation_kwargs):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def batch_generate(
        self, batch_messages: list[list], logprobs: bool = False, **generation_kwargs
    ):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    def generate_responses(self, dataset, generation_kwargs, no_batching=False):
        start_time = time.perf_counter()
        if no_batching:
            generated_outputs = []
            for conversation in dataset["conversations"]:
                _output = self.generate(conversation, **generation_kwargs)
                generated_outputs.append(_output)
        else:
            generated_outputs = self.batch_generate(
                dataset["conversations"],
                **generation_kwargs,
            )
        end_time = time.perf_counter()
        inference_time_taken = end_time - start_time

        return generated_outputs, inference_time_taken

    @abstractmethod
    def parse_outputs(self, generated_outputs, **kwargs):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )
