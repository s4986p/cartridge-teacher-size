from dataclasses import dataclass
from difflib import SequenceMatcher
import math
import re
import time

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, PreTrainedTokenizerBase, set_seed

from utils.dataset import load_longhealth_corpus, load_longhealth_questions
from utils.generation import generate_vllm, load_vllm
from utils.format import format_duration
from utils.hub import get_pending_targets, load_tensors, save_result
from utils.model import load_model, make_cartridge_cache
from utils.paths import make_result_name
from utils.prompts import LLAMA_USER_TURN

# icl은 환자 10명의 기록 전체가 prompt에 들어가, vllm 최대 길이를 llama 최대 context 길이까지 늘림
ICL_MAX_MODEL_LEN = 131072


@dataclass
class Config:
    model_name: str = "Llama-3.2-3B-Instruct"
    temperature: float = 0.3
    batch_size: int = 32
    max_generation_len: int = 1024
    seeds: tuple[int, ...] = (0, 1, 2)


def parse_longhealth_prediction(
    response: str,
    options: list[str],
) -> str | None:
    # <answer> 태그 안의 답과 가장 비슷한 선택지, 태그가 없으면 None
    # 예시) "... <answer>\nCHOP\n</answer>" → "chop"
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match is None:
        return None
    prediction = match.group(1).strip().lower()
    similarities = [SequenceMatcher(None, prediction, option.strip().lower()).ratio() for option in options]
    return options[similarities.index(max(similarities))].strip().lower()


@torch.no_grad()
def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question_ids: list[Tensor],
    max_generation_len: int,
    temperature: float,
    cartridge: dict[str, Tensor] | None,
) -> Tensor:
    cache = None if cartridge is None else make_cartridge_cache(cartridge["sink"], cartridge["cartridge"], len(question_ids))
    cache_len = 0 if cache is None else cache.get_seq_length()
    # cartridge는 맨 앞 bos를 sink로 가지고 있으므로 cartridge가 없을 때만 bos를 붙임
    bos = [torch.tensor([tokenizer.bos_token_id])] if cartridge is None else []
    question_ids = [torch.cat([*bos, ids]) for ids in question_ids]
    input_ids = pad_sequence(question_ids, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side="left")
    attention_mask = pad_sequence([torch.ones_like(ids) for ids in question_ids], batch_first=True, padding_side="left")
    # generate는 input_ids 앞에서 cache 길이만큼을 이미 처리한 token으로 보고 건너뜀
    # 따라서 cartridge 길이만큼 앞을 채우고, attention_mask는 cartridge를 보도록 1로 채움
    input_ids = pad(input_ids, (cache_len, 0), value=tokenizer.pad_token_id)
    attention_mask = pad(attention_mask, (cache_len, 0), value=1)
    output_ids = model.generate(
        input_ids=input_ids.to(model.device),
        attention_mask=attention_mask.to(model.device),
        past_key_values=cache,
        max_new_tokens=max_generation_len,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=1.0,
        top_k=0,
        pad_token_id=tokenizer.pad_token_id,
    )
    return output_ids[:, input_ids.shape[1] :]


def evaluate_responses(
    responses: list[str],
    problems: list[tuple[str, list[str], str]],
) -> dict:
    corrects = [parse_longhealth_prediction(response, options) == answer for response, (_, options, answer) in zip(responses, problems)]
    # result의 "texts" 이름은 hub에 이미 저장된 그대로
    return {"accuracy": sum(corrects) / len(problems), "corrects": corrects, "texts": responses}


def main() -> None:
    config = parse(Config)
    problems = load_longhealth_questions(use_cot=True)
    pending_targets = {seed: get_pending_targets("longhealth", config.model_name, make_result_name(seed)) for seed in config.seeds}
    print("pending", pending_targets)

    # 1. icl은 기록 전체가 prompt에 들어가 길어서 vllm으로 따로 생성
    icl_seeds = [seed for seed, targets in pending_targets.items() if f"icl/{config.model_name}" in targets]
    if icl_seeds:
        model, _ = load_vllm(config.model_name, ICL_MAX_MODEL_LEN)
        chats = [(load_longhealth_corpus(), question) for question, _, _ in problems]
        for seed in icl_seeds:
            result = evaluate_responses([output.text for output in generate_vllm(model, chats, config.max_generation_len, config.temperature, seed=seed)], problems)
            print(f"icl/{config.model_name} seed {seed} accuracy {result['accuracy']:.2%}")
            save_result(f"longhealth/icl/{config.model_name}", make_result_name(seed), result)
        model.llm_engine.engine_core.shutdown()

    # 2. base와 cartridge는 cartridge를 cache로 붙여 생성
    model, tokenizer = load_model(config.model_name)
    question_ids = [tokenizer(LLAMA_USER_TURN.format(content=question), add_special_tokens=False, return_tensors="pt")["input_ids"][0] for question, _, _ in problems]
    num_batches = math.ceil(len(problems) / config.batch_size)
    for seed, targets in pending_targets.items():
        for target in targets:
            if target.startswith("icl/"):
                continue
            cartridge = None
            if target.startswith("cartridge/"):
                cartridge = {name: tensor.to(model.device) for name, tensor in load_tensors(target, "cartridge").items()}
            set_seed(seed)
            responses = []
            start_time = time.perf_counter()
            for batch_index, start in enumerate(range(0, len(problems), config.batch_size)):
                output_ids = generate(model, tokenizer, question_ids[start : start + config.batch_size], config.max_generation_len, config.temperature, cartridge)
                responses.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
                print(f"{target} seed {seed} batch {batch_index + 1}/{num_batches} time {format_duration(time.perf_counter() - start_time)}")
                start_time = time.perf_counter()
            result = evaluate_responses(responses, problems)
            print(f"{target} seed {seed} accuracy {result['accuracy']:.2%}")
            save_result(f"longhealth/{target}", make_result_name(seed), result)


if __name__ == "__main__":
    main()
