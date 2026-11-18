from dataclasses import dataclass

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.hub import list_files, load_result
from utils.paths import make_cartridge_paths
from utils.plot import ACCURACY_YMAX, CASES, add_method_legend, get_figsize, get_panel_width, get_title, plot_method_bars, save_figure

# hub에 이미 저장된 결과 파일 이름
RESULT_NAME = "logprob"


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


def load_accuracy(
    target: str,
    files: set[str],
) -> list[float] | None:
    # seed 하나라 길이 1인 목록, 결과가 없으면 None
    if f"longhealth_logprob/{target}/{RESULT_NAME}.json" not in files:
        return None
    return [load_result(f"longhealth_logprob/{target}", RESULT_NAME)["accuracy"]]


def main() -> None:
    config = parse(Config)
    files = list_files()
    reference_values = {method: load_accuracy(f"{method}/{config.model_name}", files) for method in ["base", "icl"]}
    cartridge_paths = make_cartridge_paths(config.model_name, config.teacher_name, config.cartridge_len, config.learning_rate, config.batch_size, config.num_epochs, config.cartridge_seed)
    cartridge_values = {case: load_accuracy(path, files) for case, path in cartridge_paths.items()}

    fig, ax = plt.subplots(figsize=get_figsize([get_panel_width(len(CASES), 2)], 1), layout="constrained")
    plot_method_bars(ax, reference_values, cartridge_values)
    ax.set_ylim(0, ACCURACY_YMAX)
    ax.set_ylabel("정확도")

    fig.suptitle(get_title(config.model_name, "logprob 채점"))
    add_method_legend(fig)
    save_figure(fig, config.model_name, "longhealth_logprob_scoring")


if __name__ == "__main__":
    main()
