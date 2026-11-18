import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, PreTrainedModel, PreTrainedTokenizerBase

# model_name은 hub 경로에 쓰는 이름이라 org를 붙여 불러옴
MODEL_ORG = "meta-llama"


def load_model(
    model_name: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    # llama tokenizer는 pad token이 없어 generate에서 오류가 나기 때문에 eos를 pad로 씀
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_ORG}/{model_name}")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(f"{MODEL_ORG}/{model_name}", dtype=torch.bfloat16, device_map="auto")
    model.requires_grad_(False)
    return model, tokenizer


def make_cartridge_cache(
    sink: Tensor,
    cartridge: Tensor,
    batch_size: int,
) -> DynamicCache:
    # 맨 앞 bos인 sink 뒤에 cartridge를 이어 batch 크기만큼 복사
    key_value = torch.cat([sink, cartridge.to(sink.dtype)], dim=-2)
    return DynamicCache(ddp_cache_data=key_value.repeat_interleave(batch_size, dim=2))
