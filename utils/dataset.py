import requests

from utils.prompts import NOTE

LONGHEALTH_URL = "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"
# cartridges 레포처럼 앞에서부터 환자 10명만 사용
NUM_PATIENTS = 10


def load_longhealth_patients() -> list[tuple[str, dict]]:
    return sorted(requests.get(LONGHEALTH_URL).json().items())[:NUM_PATIENTS]


def get_notes(
    patient: dict,
) -> list[str]:
    # 예시) {"text_0": "..."} → ["<text_0>\n...\n</text_0>"]
    return [NOTE.format(note_id=note_id, note=note) for note_id, note in patient["texts"].items()]
