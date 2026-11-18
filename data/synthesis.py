from collections import defaultdict
from dataclasses import dataclass
import math
import random
import time

from simple_parsing import parse
import torch

from utils.dataset import get_notes, load_longhealth_patients
from utils.evaluation import generate_vllm, load_vllm
from utils.format import format_duration
from utils.hub import save_synthesis
from utils.paths import make_synthesis_path
from utils.prompts import COT_PROMPTS, FORMATS, RECORD, SEED_PROMPTS, SYSTEM_PROMPT


@dataclass
class Config:
    teacher_name: str = "Llama-3.2-3B-Instruct"
    num_samples: int = 16384
    cot_ratio: float = 0.75
    max_question_len: int = 512
    question_temperature: float = 0.6
    max_response_len: int = 1024
    response_temperature: float = 0.0
    topk: int = 20
    batch_size: int = 1024


def load_longhealth_contexts() -> list[str]:
    # 환자 정보와 note 하나씩을 묶어 context로 씀
    contexts = []
    for patient_id, patient in load_longhealth_patients():
        notes = get_notes(patient)
        contexts.extend(RECORD.format(patient_id=patient_id, num_notes=len(notes), notes=note, **patient) for note in notes)
    return contexts


def main() -> None:
    config = parse(Config)
    contexts = load_longhealth_contexts()
    model, tokenizer = load_vllm(config.teacher_name)

    # 1. synthesis마다 context, seed prompt, cot prompt를 무작위로 고름, cot prompt는 cot_ratio 확률로만 붙임
    prompts = []
    for _ in range(config.num_samples):
        context = random.choice(contexts)
        seed_prompt = random.choice(random.choice(SEED_PROMPTS)).format(format=random.choice(FORMATS))
        cot_prompt = random.choice(COT_PROMPTS) if random.random() < config.cot_ratio else None
        prompts.append((context, seed_prompt, cot_prompt))

    rows = []
    tensors = defaultdict(list)
    num_batches = math.ceil(len(prompts) / config.batch_size)
    start_time = time.perf_counter()
    for batch_index, start in enumerate(range(0, len(prompts), config.batch_size)):
        batch = prompts[start : start + config.batch_size]

        # 2. teacher가 seed prompt로 question을 쓰고, 같은 context로 question에 response를 씀
        chats = [(SYSTEM_PROMPT.format(context=context), seed_prompt) for context, seed_prompt, _ in batch]
        questions = [output.text for output in generate_vllm(model, chats, config.max_question_len, config.question_temperature)]
        contents = [f"{question}\n\n{cot_prompt}" if cot_prompt else question for (_, _, cot_prompt), question in zip(batch, questions)]
        chats = [(SYSTEM_PROMPT.format(context=context), content) for (context, _, _), content in zip(batch, contents)]
        outputs = generate_vllm(model, chats, config.max_response_len, config.response_temperature, topk=config.topk)

        # 3. response token마다 teacher top-k logprob 저장, vllm은 샘플링한 token의 logprob을 top-k 앞에 붙이므로 뺌
        # rows의 "answer" 이름은 hub에 이미 저장된 그대로
        rows.extend({"context": context, "question": question, "cot": cot_prompt, "answer": tokenizer.decode(output.token_ids)} for (context, _, cot_prompt), question, output in zip(batch, questions, outputs))
        tensors["answer_ids"].append(torch.cat([torch.tensor(output.token_ids) for output in outputs]))
        tensors["topk_ids"].append(torch.cat([torch.tensor(output.logprobs.token_ids).view(-1, config.topk + 1)[:, 1:] for output in outputs]))
        tensors["topk_logprobs"].append(torch.cat([torch.tensor(output.logprobs.logprobs).view(-1, config.topk + 1)[:, 1:] for output in outputs]))
        tensors["answer_lens"].append(torch.tensor([len(output.token_ids) for output in outputs]))
        print(f"batch {batch_index + 1}/{num_batches} time {format_duration(time.perf_counter() - start_time)}")
        start_time = time.perf_counter()

    save_synthesis(make_synthesis_path(config.teacher_name, config.num_samples, config.cot_ratio), rows, {name: torch.cat(values) for name, values in tensors.items()})


if __name__ == "__main__":
    main()
