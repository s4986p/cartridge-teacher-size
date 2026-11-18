from pathlib import Path
from statistics import mean, stdev

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

MODEL_NAMES = {"Llama-3.2-3B-Instruct": "llama-3.2-3b-instruct", "Llama-3.1-8B-Instruct": "llama-3.1-8b-instruct"}
METHOD_COLORS = {"base": "#a3a29b", "icl": "#1baf7a", "self": "#2a78d6", "cross": "#eb6834"}
BAR_INCHES = 0.45
BAR_GAP = 1
PANEL_HEIGHT = 3.0
PANEL_MARGIN = 0.9
# case가 적어도 제목과 범례가 잘리지 않는 너비
MIN_FIGURE_WIDTH = 9.0
# 정확도와 비율은 plot끼리 비교하기 쉽게 y축 고정, 막대 위 값이 잘리지 않게 여유를 둠
ACCURACY_YMAX = 0.7
RATIO_YMAX = 0.4
# 막대 위 값 글자 크기와 막대에서 띄우는 point 수
VALUE_FONTSIZE = 8
VALUE_OFFSET = 3
# 기준선 회색
NEUTRAL_COLOR = "#808080"
BASELINE_WIDTH = 0.8

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "axes.unicode_minus": False,
    "figure.titlesize": 13,
    "figure.constrained_layout.w_pad": 0.08,
    "figure.constrained_layout.h_pad": 0.08,
    "figure.constrained_layout.wspace": 0.08,
    "figure.constrained_layout.hspace": 0.08,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": "#e8e7e3",
    "grid.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "savefig.dpi": 200,
})


def get_title(
    model_name: str,
    experiment_name: str,
) -> str:
    # 예시) "llama-3.2-3b-instruct longhealth teacher별 정확도"
    return f"{MODEL_NAMES[model_name]} longhealth {experiment_name}"


def get_panel_width(
    num_cases: int,
    num_series: int,
) -> float:
    # case마다 series 막대와 간격
    return BAR_INCHES * num_cases * (num_series + BAR_GAP)


def get_figsize(
    panel_widths: list[float],
    num_rows: int,
) -> tuple[float, float]:
    # panel을 가로로, 묶음을 세로로 놓음
    return max(sum(panel_widths) + PANEL_MARGIN * len(panel_widths), MIN_FIGURE_WIDTH), PANEL_HEIGHT * num_rows + 1


def format_value(
    values: list[float],
    separator: str = "\n",
) -> str:
    # seed 평균, seed가 여럿이면 separator 뒤에 seed 사이 표준편차
    # 예시) [0.47, 0.465, 0.48] → "47.2%\n±0.8%p"
    text = f"{mean(values):.1%}"
    return text if len(values) == 1 else f"{text}{separator}±{stdev(values):.1%}p"


def plot_seed_bars(
    ax: Axes,
    cases: list[str],
    bars: list[list[tuple[list[float], str]]],
) -> None:
    # case마다 막대를 나란히 그림, 막대는 seed 평균이고 위에 format_value로 값 표시
    # bars는 case마다 (seed별 값, 색) 목록, case마다 막대 수가 달라도 가운데 정렬
    # 예시) cases ["base", "65536개"], bars [[(base 값, 회색)], [(self 값, 파랑), (cross 값, 주황)]]
    case_width = max(len(case_bars) for case_bars in bars) + BAR_GAP
    for case_index, case_bars in enumerate(bars):
        for bar_index, (values, color) in enumerate(case_bars):
            x = case_index * case_width + bar_index - (len(case_bars) - 1) / 2
            ax.bar(x, mean(values), width=1, color=color)
            ax.annotate(format_value(values), (x, mean(values)), xytext=(0, VALUE_OFFSET), textcoords="offset points", ha="center", va="bottom", fontsize=VALUE_FONTSIZE)
    ax.axhline(0, color=NEUTRAL_COLOR, linewidth=BASELINE_WIDTH)
    ax.set_xticks([case_index * case_width for case_index in range(len(cases))], cases)
    ax.set_xlim(-case_width / 2, (len(cases) - 0.5) * case_width)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))


def save_figure(
    fig: Figure,
    model_name: str,
    file_name: str,
) -> None:
    path = Path(f"figures/{model_name}/{file_name}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"saved {path}")
