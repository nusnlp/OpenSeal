import asyncio
import os
import pandas as pd

import litellm
from openai import OpenAI

from base_logger import get_logger
from serving.litellm_serving import LiteLLMServing
import tiktoken

logger = get_logger(__name__)
special_token_map = {
    "gpt-3.5": {
        "<|im_start|>": 100264,
        "<|im_end|>": 100265,
    },
    "gpt-4": {
        "<|im_start|>": 200264,
        "<|im_sep|>": 200266,
        "<|im_end|>": 200265,
    },
}

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    models = client.models.list()
    OPENAI_GPT_MODELS = {x.id for x in models.data if "gpt" in x.id}
    OPENAI_O1_MODELS = {x.id for x in models.data if "o1" in x.id}
    OPENAI_MODELS = OPENAI_GPT_MODELS.union(OPENAI_O1_MODELS)
except:
    OPENAI_MODELS = set()
    logger.warning(
        "Unable to get list of OpenAI models. Please check your OpenAI API key."
    )


class OpenAIServing(LiteLLMServing):
    def __init__(
        self,
        model: str,
        base_url: str = None,
        api_key: str = None,
        is_base_model: bool = False,
        num_retries: int = 5,
    ):
        super().__init__(
            model=model, base_url=base_url, api_key=api_key, is_base_model=is_base_model
        )
        assert model in OPENAI_MODELS, f"Invalid OpenAI model name: {model}"
        self.num_retries = num_retries
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")

        self.tokenizer = tiktoken.encoding_for_model(self.model)

    def apply_chat_template(self, message, add_generation_prompt=True):
        output = []
        if self.model.lower().startswith("gpt-4"):
            # for gpt-4 and gpt-4o
            for m in message:
                output.extend(
                    [
                        "<|im_start|>",
                        m["role"],
                        "<|im_sep|>",
                        m["content"],
                        "<|im_end|>",
                    ]
                )

            if add_generation_prompt:
                output.extend(["<|im_start|>", "assistant", "<|im_sep|>"])
        elif self.model.lower().startswith("gpt-3.5"):
            for m in message:
                output.extend(
                    ["<|im_start|>", m["role"] + "\n", m["content"], "<|im_end|>"]
                )

            if add_generation_prompt:
                output.extend(["<|im_start|>", "assistant\n"])
        return output

    def tokenize(self, message, add_generation_prompt=True):
        chat_input = self.apply_chat_template(
            message, add_generation_prompt=add_generation_prompt
        )

        # hack to solve token
        tokens = []
        for k, v in special_token_map.items():
            if k in self.model.lower():
                token_map = v
                break

        for input in chat_input:
            if input in token_map.keys():
                tokens.append(token_map[input])
            else:
                tokens.extend(self.tokenizer.encode(input))

        return tokens

    def prepare_llm_batches(
        self, llm_batch_file_path, model, conversations, **generation_kwargs
    ):
        batches = []
        id = os.path.splitext(os.path.split(llm_batch_file_path)[-1])[0]

        for i, convo in enumerate(conversations):
            # prepare body of batch to send to OpenAI API
            body = (
                generation_kwargs.copy()
            )  # must copy to ensure that the values don't change
            body["model"] = model
            body["messages"] = convo

            output = {
                "custom_id": f"{id}_{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            batches.append(output)
        df = pd.DataFrame(batches)
        df.to_json(llm_batch_file_path, orient="records", lines=True, force_ascii=False)

    # def encode_prompts
    async def abatch_generate(
        self,
        file_path: str,
        output_file_path: str,
        custom_llm_provider: str = "openai",
        sleep_time: int = 10,
    ):
        file_obj = await litellm.acreate_file(
            file=open(file_path, "rb"),
            purpose="batch",
            custom_llm_provider=custom_llm_provider,
        )

        await asyncio.sleep(10)
        batch_input_file_id = file_obj.id
        assert (
            batch_input_file_id is not None
        ), "Failed to create input file, expected a non null file_id but got {batch_input_file_id}"

        create_batch_response = await litellm.acreate_batch(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=batch_input_file_id,
            custom_llm_provider=custom_llm_provider,
        )
        logger.info("Prompts sent via OpenAI batch API")

        is_batch_completed = False
        logger.info("Waiting for OpenAI batch to complete...")
        counter = 1
        while is_batch_completed == False:
            await asyncio.sleep(sleep_time)

            retrieved_batch = await litellm.aretrieve_batch(
                batch_id=create_batch_response.id,
                custom_llm_provider=custom_llm_provider,
            )
            is_batch_completed = retrieved_batch.status == "completed"
            logger.info(
                "Still waiting (%ds has elapsed)...",
                counter * sleep_time,
            )
            counter += 1
        logger.info("OpenAI batch is completed")

        file_content = await litellm.afile_content(
            file_id=retrieved_batch.output_file_id, custom_llm_provider="openai"
        )
        file_content.write_to_file(output_file_path)

        return file_content.read()

    def generate_openai_batched_responses(
        self, dataset, generation_kwargs, output_dir, model_name, task_name, lang
    ):
        batch_filepath = os.path.join(
            output_dir,
            os.path.basename(model_name),
            "inference",
            f"{os.path.basename(model_name)}_{task_name}_{lang}_batch.jsonl",
        )
        self.prepare_llm_batches(
            batch_filepath,
            model_name,
            dataset["conversations"],
            **generation_kwargs,
        )
        batch_output_filepath = os.path.join(
            output_dir,
            os.path.basename(model_name),
            "inference",
            f"{os.path.basename(model_name)}_{task_name}_{lang}_batch_response.jsonl",
        )
        asyncio.run(self.abatch_generate(batch_filepath, batch_output_filepath))
        generated_outputs = pd.read_json(batch_output_filepath, lines=True).to_dict(
            "records"
        )
        return generated_outputs

    def get_response(self, output):
        return output["response"]["body"]["choices"][0]["message"]["content"]


if __name__ == "__main__":
    openai_model = OpenAIServing("gpt-4-0613")
    messages = [
        {
            "role": "system",
            "content": """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. The language used by the AI assistants should also be the same as that used in the user question. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.""",
        },
        {
            "role": "user",
            "content": """[User Question]
{question}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]""".format(
                question="Bayangkan Anda sedang menulis postingan blog untuk membandingkan dua model smartphone populer. Buatlah kerangka untuk postingan blog tersebut, termasuk poin-poin kunci dan subjudul untuk membandingkan dan membedakan fitur, kinerja, dan pengalaman pengguna dari kedua model tersebut. Harap jawab dalam kurang dari 200 kata.",
                answer_a="##  [Nama Smartphone 1] vs [Nama Smartphone 2]: Pertempuran Smartphone Kelas Atas\n\n**Pendahuluan:**\n\n*  Perkenalan singkat tentang kedua smartphone dan target pasar mereka.\n*  Menjelaskan tujuan postingan blog: membandingkan dan membedakan kedua model.\n\n**Desain dan Layar:**\n\n*  Membandingkan ukuran, berat, material, dan estetika desain.\n*  Membahas jenis layar (AMOLED, LCD), resolusi, refresh rate, dan kualitas visual.\n\n**Performa dan Kamera:**\n\n*  Membandingkan prosesor, RAM, penyimpanan, dan benchmark kinerja.\n*  Membahas spesifikasi kamera (resolusi, aperture, fitur), kualitas foto dan video di berbagai kondisi.\n\n**Fitur dan Software:**\n\n*  Membandingkan sistem operasi, antarmuka pengguna, fitur unik (misalnya, NFC, pengenalan wajah), dan dukungan software.\n\n**Baterai dan Ketahanan:**\n\n*  Membandingkan kapasitas baterai, pengisian daya cepat, dan waktu penggunaan.\n*  Membahas ketahanan air dan debu (jika ada).\n\n**Harga dan Kesimpulan:**\n\n*  Membandingkan harga kedua smartphone.\n*  Menyimpulkan smartphone mana yang lebih unggul berdasarkan kebutuhan dan preferensi pengguna.",
                answer_b="Judul: Perbandingan Antara Smartphone Model A dan Model B\n\nI. Pendahuluan\n- Pengenalan tentang kedua model smartphone yang akan dibandingkan\n- Tujuan dari perbandingan ini\n\nII. Desain\n- Material dan desain fisik dari masing-masing model\n- Ukuran layar dan resolusi yang dimiliki\n- Bobot dan ketebalan smartphone\n\nIII. Fitur\n- Spesifikasi kamera, termasuk resolusi dan fitur tambahan\n- Kapasitas baterai dan teknologi pengisian daya\n- Keamanan dan privasi, seperti sensor sidik jari atau pengenalan wajah\n\nIV. Kinerja\n- Prosesor dan RAM yang digunakan\n- Kapasitas penyimpanan internal dan kemampuan ekspansi\n- Performa dalam penggunaan sehari-hari dan multitasking\n\nV. Pengalaman Pengguna\n- Antarmuka pengguna yang digunakan\n- Kualitas suara dan fitur multimedia\n- Ketersediaan update sistem operasi dan dukungan purna jual\n\nVI. Kesimpulan\n- Ringkasan perbandingan antara kedua model smartphone\n- Rekomendasi untuk konsumen berdasarkan kebutuhan dan preferensi mereka",
            ),
        },
    ]

    response = openai_model.generate(messages)
    print(response.choices[0].message.content)
