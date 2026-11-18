# self와 cross를 비교하는 (synthesis 수, cot 비율)
SYNTHESIS_CASES = [(16384, 0.75), (16384, 0.2), (65536, 0.2)]


def make_synthesis_path(
    teacher_name: str,
    num_samples: int,
    cot_ratio: float,
) -> str:
    return f"synthesis/{teacher_name}/num_samples={num_samples},cot_ratio={cot_ratio:g}"


def make_cartridge_path(
    model_name: str,
    teacher_name: str,
    num_samples: int,
    cot_ratio: float,
    cartridge_len: int,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    seed: int,
) -> str:
    # 예시) "cartridge/Llama-3.2-3B-Instruct/teacher=Llama-3.1-8B-Instruct,num_samples=65536,cot_ratio=0.2,cartridge_len=2048,learning_rate=0.02,batch_size=64,num_epochs=2,seed=0"
    return f"cartridge/{model_name}/teacher={teacher_name},num_samples={num_samples},cot_ratio={cot_ratio:g},cartridge_len={cartridge_len},learning_rate={learning_rate:g},batch_size={batch_size},num_epochs={num_epochs},seed={seed}"


def make_result_name(
    seed: int,
) -> str:
    return f"seed={seed}"


def make_cartridge_paths(
    model_name: str,
    teacher_name: str,
    cartridge_len: int,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    seed: int,
) -> dict[tuple[str, int, float], str]:
    # (self 또는 cross, synthesis 수, cot 비율)마다 cartridge 경로
    # self는 student 자신이 teacher, cross는 teacher_name이 teacher
    paths = {}
    for method, method_teacher_name in [("self", model_name), ("cross", teacher_name)]:
        for num_samples, cot_ratio in SYNTHESIS_CASES:
            paths[method, num_samples, cot_ratio] = make_cartridge_path(model_name, method_teacher_name, num_samples, cot_ratio, cartridge_len, learning_rate, batch_size, num_epochs, seed)
    return paths
