from vllm import LLM, SamplingParams  # type: ignore
from vllm.outputs import CompletionOutput  # type: ignore
from vllm.tokenizers import TokenizerLike  # type: ignore

from utils.model import MODEL_ORG
from utils.prompts import DATE_STRING

# synthesis는 context와 question, response가 이 안에 들어감
MAX_MODEL_LEN = 16384


def load_vllm(
    model_name: str,
    max_model_len: int = MAX_MODEL_LEN,
) -> tuple[LLM, TokenizerLike]:
    model = LLM(f"{MODEL_ORG}/{model_name}", dtype="bfloat16", max_model_len=max_model_len)
    return model, model.get_tokenizer()


def generate_vllm(
    model: LLM,
    chats: list[tuple[str, str]],
    max_generation_len: int,
    temperature: float,
    topk: int | None = None,
    seed: int | None = None,
) -> list[CompletionOutput]:
    # chats는 (system, user) 쌍, topk가 있으면 생성한 token마다 top-k logprob도 받음
    messages = [[{"role": "system", "content": system}, {"role": "user", "content": user}] for system, user in chats]
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_generation_len, logprobs=topk, flat_logprobs=True, seed=seed)
    outputs = model.chat(messages, sampling_params, use_tqdm=False, chat_template_kwargs={"date_string": DATE_STRING})
    return [output.outputs[0] for output in outputs]
