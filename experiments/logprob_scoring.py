from dataclasses import dataclass
import time

from simple_parsing import parse
import torch
from torch.nn.functional import log_softmax
from transformers import DynamicCache

from utils.dataset import load_longhealth_corpus, load_longhealth_questions
from utils.format import format_duration
from utils.hub import get_pending_targets, load_tensors, save_result
from utils.model import load_model
from utils.prompts import DATE_STRING, LLAMA_USER_TURN

# hub에 이미 저장된 결과 파일 이름
RESULT_NAME = "logprob"
# 이 문제 수마다 진행 상황 출력
LOG_PROBLEMS = 20


@dataclass
class Config:
    model_name: str = "Llama-3.2-3B-Instruct"


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    experiment_name = "longhealth_logprob"
    targets = get_pending_targets(experiment_name, config.model_name, RESULT_NAME)
    print("pending", targets)
    problems = load_longhealth_questions(use_cot=False)
    model, tokenizer = load_model(config.model_name)
    question_ids = [tokenizer(LLAMA_USER_TURN.format(content=question), add_special_tokens=False, return_tensors="pt")["input_ids"][0] for question, _, _ in problems]
    answer_prefix_ids = tokenizer("<answer>\n", add_special_tokens=False, return_tensors="pt")["input_ids"][0]

    for target in targets:
        # 1. icl은 기록 전체, cartridge는 sink와 cartridge를 cache로 붙임
        # icl과 cartridge는 cache 맨 앞에 bos가 있으므로 base만 bos를 붙임
        key_value = None
        bos_ids = []
        if target.startswith("icl/"):
            corpus_ids = tokenizer.apply_chat_template([{"role": "system", "content": load_longhealth_corpus()}], date_string=DATE_STRING, return_tensors="pt")["input_ids"]
            cache = DynamicCache()
            model.model(input_ids=corpus_ids.to(model.device), past_key_values=cache)
            key_value = torch.stack([torch.stack([layer.keys, layer.values]) for layer in cache.layers])
        elif target.startswith("cartridge/"):
            tensors = load_tensors(target, "cartridge")
            key_value = torch.cat([tensors["sink"], tensors["cartridge"].to(tensors["sink"].dtype)], dim=-2).to(model.device)
        else:
            bos_ids = [torch.tensor([tokenizer.bos_token_id])]

        # 2. 선택지마다 "<answer>\n" 뒤 선택지 token의 평균 logprob을 구하고 가장 높은 선택지를 답으로 봄
        corrects = []
        start_time = time.perf_counter()
        for problem_index, ((_, options, answer), problem_question_ids) in enumerate(zip(problems, question_ids)):
            scores = []
            for option in options:
                option_ids = tokenizer(option.strip(), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                input_ids = torch.cat([*bos_ids, problem_question_ids, answer_prefix_ids, option_ids])[None, :].to(model.device)
                cache = None if key_value is None else DynamicCache(ddp_cache_data=key_value)
                # t의 hidden이 t+1 token을 예측하기 때문에 선택지 바로 앞 token부터 끝에서 두 번째 token까지 유지
                hiddens = model.model(input_ids=input_ids, past_key_values=cache).last_hidden_state[0, -len(option_ids) - 1 : -1]
                logprobs = log_softmax(model.lm_head(hiddens).float(), dim=-1)
                scores.append(logprobs.gather(-1, option_ids[:, None].to(model.device)).mean().item())
            corrects.append(options[scores.index(max(scores))].strip().lower() == answer)
            if (problem_index + 1) % LOG_PROBLEMS == 0:
                print(f"{target} problem {problem_index + 1}/{len(problems)} accuracy {sum(corrects) / len(corrects):.2%} time {format_duration(time.perf_counter() - start_time)}")
                start_time = time.perf_counter()

        accuracy = sum(corrects) / len(problems)
        print(f"{target} accuracy {accuracy:.2%}")
        save_result(f"{experiment_name}/{target}", RESULT_NAME, {"accuracy": accuracy, "corrects": corrects})


if __name__ == "__main__":
    main()
