from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time

from datasets import load_dataset
from safetensors.torch import load_file
from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn.functional import log_softmax
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import DynamicCache, PreTrainedTokenizerBase, set_seed

from utils.format import format_duration, format_memory
from utils.hub import download_file, save_tensors
from utils.model import load_model, make_cartridge_cache
from utils.paths import make_cartridge_path, make_synthesis_path
from utils.prompts import DATE_STRING, LLAMA_USER_TURN

# 이 batch 수마다 loss 평균 출력
LOG_BATCHES = 16


@dataclass
class Config:
    model_name: str = "Llama-3.2-3B-Instruct"
    teacher_name: str = "Llama-3.1-8B-Instruct"
    # synthesis 데이터
    num_samples: int = 16384
    cot_ratio: float = 0.75
    # cartridge 학습
    cartridge_len: int = 2048
    learning_rate: float = 2e-2
    batch_size: int = 64
    micro_batch_size: int = 4
    num_epochs: int = 2
    seed: int = 0


def load_wikitext(
    tokenizer: PreTrainedTokenizerBase,
    cartridge_len: int,
) -> Tensor:
    # cartridge_len token이 넘을 때까지 wikitext-2를 이어 붙여 system turn으로 만듦
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = ""
    for row in dataset:
        text += row["text"]
        if len(tokenizer(text)["input_ids"]) > cartridge_len:
            break
    # cartridge_len개로 자르되 system turn을 닫는 마지막 token은 남김
    input_ids = tokenizer.apply_chat_template([{"role": "system", "content": text}], date_string=DATE_STRING, return_tensors="pt")["input_ids"][0]
    return torch.cat([input_ids[:cartridge_len], input_ids[-1:]])


def load_synthesis(
    hub_path: str,
    tokenizer: PreTrainedTokenizerBase,
) -> list[tuple[Tensor, Tensor, Tensor, Tensor]]:
    # synthesis마다 (question ids, response ids, teacher top-k ids, teacher top-k logprobs)
    rows = json.loads(Path(download_file(f"{hub_path}/rows.json")).read_text())
    tensors = load_file(download_file(f"{hub_path}/logprobs.safetensors"))
    response_lens = tensors["answer_lens"].tolist()
    response_ids = tensors["answer_ids"].split(response_lens)
    topk_ids = tensors["topk_ids"].split(response_lens)
    topk_logprobs = tensors["topk_logprobs"].split(response_lens)

    # student는 context 없이 cartridge만 붙이므로 question만 user turn으로 넣음
    contents = [f"{row['question']}\n\n{row['cot']}" if row["cot"] else row["question"] for row in rows]
    turns = [LLAMA_USER_TURN.format(content=content.strip()) for content in contents]
    question_ids = [tokenizer(turn, add_special_tokens=False, return_tensors="pt")["input_ids"][0] for turn in turns]
    return list(zip(question_ids, response_ids, topk_ids, topk_logprobs))


def main() -> None:
    config = parse(Config)
    set_seed(config.seed)
    model, tokenizer = load_model(config.model_name)
    samples = load_synthesis(make_synthesis_path(config.teacher_name, config.num_samples, config.cot_ratio), tokenizer)

    # 1. cartridge는 문서와 상관없는 text의 kv cache로 초기화, 맨 앞 bos는 sink로 고정하고 나머지만 학습
    cache = DynamicCache()
    model.model(input_ids=load_wikitext(tokenizer, config.cartridge_len)[None, :].to(model.device), past_key_values=cache)
    key_value = torch.stack([torch.stack([layer.keys, layer.values]) for layer in cache.layers])
    sink = key_value[..., :1, :]
    cartridge = Parameter(key_value[..., 1:, :].float())
    optimizer = AdamW([cartridge], lr=config.learning_rate, weight_decay=0)

    # 2. cartridge만 붙인 student가 response token마다 teacher top-k 분포와의 cross entropy를 줄임
    num_batches = math.ceil(len(samples) / config.batch_size)
    losses = []
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(config.num_epochs):
        random.shuffle(samples)
        for batch_index, start in enumerate(range(0, len(samples), config.batch_size)):
            batch = samples[start : start + config.batch_size]
            num_response_tokens = sum(len(response_ids) for _, response_ids, _, _ in batch)
            batch_loss = 0.0
            # 메모리 때문에 micro_batch_size씩 나눠 gradient를 모음
            for micro_start in range(0, len(batch), config.micro_batch_size):
                micro_batch = batch[micro_start : micro_start + config.micro_batch_size]
                input_ids = pad_sequence([torch.cat([question_ids, response_ids]) for question_ids, response_ids, _, _ in micro_batch], batch_first=True).to(model.device)
                hiddens = model.model(input_ids=input_ids, past_key_values=make_cartridge_cache(sink, cartridge, len(micro_batch))).last_hidden_state
                # t의 hidden이 t+1 token을 예측하기 때문에 마지막 question token부터 끝에서 두 번째 response token까지 유지
                response_hiddens = [hiddens[index, len(question_ids) - 1 : len(question_ids) + len(response_ids) - 1] for index, (question_ids, response_ids, _, _) in enumerate(micro_batch)]
                logprobs = log_softmax(model.lm_head(torch.cat(response_hiddens)).float(), dim=-1)
                topk_ids = torch.cat([sample_topk_ids for _, _, sample_topk_ids, _ in micro_batch]).to(model.device)
                topk_logprobs = torch.cat([sample_topk_logprobs for _, _, _, sample_topk_logprobs in micro_batch]).to(model.device)
                loss = -(topk_logprobs.exp() * logprobs.gather(-1, topk_ids)).sum() / num_response_tokens
                loss.backward()
                batch_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()
            losses.append(batch_loss)
            if (batch_index + 1) % LOG_BATCHES == 0:
                print(f"epoch {epoch + 1}/{config.num_epochs} batch {batch_index + 1}/{num_batches} loss {sum(losses) / len(losses):.4f} time {format_duration(time.perf_counter() - start_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
                losses = []
                start_time = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()

    hub_path = make_cartridge_path(config.model_name, config.teacher_name, config.num_samples, config.cot_ratio, config.cartridge_len, config.learning_rate, config.batch_size, config.num_epochs, config.seed)
    save_tensors(hub_path, "cartridge", {"sink": sink, "cartridge": cartridge})


if __name__ == "__main__":
    main()
