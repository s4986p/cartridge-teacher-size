from dataclasses import dataclass
import re
from statistics import mean

import matplotlib.pyplot as plt
from simple_parsing import parse
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from utils.hub import list_files, load_result
from utils.model import MODEL_ORG
from utils.paths import make_cartridge_paths, make_result_name
from utils.plot import ACCURACY_YMAX, CASES, RATIO_YMAX, add_method_legend, get_figsize, get_panel_width, get_title, plot_method_bars, save_figure

# 반복: 끝에서 LOOP_TAIL_LEN번째부터 LOOP_PATTERN_LEN 글자가 그 앞에 MIN_LOOP_COUNT번 이상 나옴
LOOP_TAIL_LEN = 200
LOOP_PATTERN_LEN = 50
MIN_LOOP_COUNT = 5
PANEL_ROWS = 2


@dataclass
class Config:
    model_name: str = "Llama-3.2-3B-Instruct"
    teacher_name: str = "Llama-3.1-8B-Instruct"
    # cartridge 학습
    cartridge_len: int = 2048
    learning_rate: float = 2e-2
    batch_size: int = 64
    num_epochs: int = 2
    cartridge_seed: int = 0
    max_generation_len: int = 1024
    seeds: tuple[int, ...] = (0, 1, 2)


def has_answer_tag(
    response: str,
) -> bool:
    return re.search(r"<answer>(.*?)</answer>", response, re.DOTALL) is not None


def is_loop(
    response: str,
) -> bool:
    return len(response) > LOOP_TAIL_LEN and response[:-LOOP_TAIL_LEN].count(response[-LOOP_TAIL_LEN : -LOOP_TAIL_LEN + LOOP_PATTERN_LEN]) >= MIN_LOOP_COUNT


def get_metrics(
    target: str,
    files: set[str],
    tokenizer: PreTrainedTokenizerBase,
    max_generation_len: int,
    seeds: tuple[int, ...],
) -> dict[str, list[float]] | None:
    # seed마다 정확도, 태그 없음, 잘림, 반복 비율, 결과가 없으면 None
    if f"longhealth/{target}/{make_result_name(seeds[0])}.json" not in files:
        return None
    metrics = {"정확도": [], "태그 없음": [], "잘림": [], "반복": []}
    for seed in seeds:
        result = load_result(f"longhealth/{target}", make_result_name(seed))
        responses = result["texts"]
        metrics["정확도"].append(result["accuracy"])
        metrics["태그 없음"].append(mean(not has_answer_tag(response) for response in responses))
        # 잘림은 최대 생성 길이에 닿고 틀린 문제
        metrics["잘림"].append(mean(len(tokenizer(response, add_special_tokens=False)["input_ids"]) >= max_generation_len and not correct for response, correct in zip(responses, result["corrects"])))
        metrics["반복"].append(mean(is_loop(response) for response in responses))
    return metrics


def main() -> None:
    config = parse(Config)
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_ORG}/{config.model_name}")
    files = list_files()

    reference_metrics = {method: get_metrics(f"{method}/{config.model_name}", files, tokenizer, config.max_generation_len, config.seeds) for method in ["base", "icl"]}
    cartridge_paths = make_cartridge_paths(config.model_name, config.teacher_name, config.cartridge_len, config.learning_rate, config.batch_size, config.num_epochs, config.cartridge_seed)
    cartridge_metrics = {case: get_metrics(path, files, tokenizer, config.max_generation_len, config.seeds) for case, path in cartridge_paths.items()}

    # panel 4개를 2x2로 놓음
    panel_names = list(reference_metrics["base"])
    fig, axes = plt.subplots(PANEL_ROWS, len(panel_names) // PANEL_ROWS, figsize=get_figsize([get_panel_width(len(CASES), 2)] * (len(panel_names) // PANEL_ROWS), PANEL_ROWS), layout="constrained")
    for ax, panel_name in zip(axes.flat, panel_names):
        reference_values = {method: metrics[panel_name] for method, metrics in reference_metrics.items()}
        cartridge_values = {case: None if metrics is None else metrics[panel_name] for case, metrics in cartridge_metrics.items()}
        plot_method_bars(ax, reference_values, cartridge_values)
        ax.set_ylim(0, ACCURACY_YMAX if panel_name == "정확도" else RATIO_YMAX)
        ax.set_ylabel(panel_name)

    fig.suptitle(get_title(config.model_name, "teacher별 정확도"))
    add_method_legend(fig)
    save_figure(fig, config.model_name, "longhealth_teacher_accuracy")


if __name__ == "__main__":
    main()
