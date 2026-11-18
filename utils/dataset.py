import requests

from utils.prompts import CORPUS, LONGHEALTH_NO_COT_QUESTION, LONGHEALTH_QUESTION, NOTE, PATIENT

LONGHEALTH_URL = "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"
# cartridges 레포처럼 앞에서부터 환자 10명만 사용
NUM_PATIENTS = 10
OPTION_LETTERS = "abcde"


def load_longhealth_patients() -> list[tuple[str, dict]]:
    return sorted(requests.get(LONGHEALTH_URL).json().items())[:NUM_PATIENTS]


def get_notes(
    patient: dict,
) -> list[str]:
    # 예시) {"text_0": "..."} → ["<text_0>\n...\n</text_0>"]
    return [NOTE.format(note_id=note_id, note=note) for note_id, note in patient["texts"].items()]


def load_longhealth_corpus() -> str:
    records = []
    for patient_id, patient in load_longhealth_patients():
        notes = get_notes(patient)
        records.append(PATIENT.format(patient_id=patient_id, num_notes=len(notes), notes="\n".join(notes), **patient))
    return CORPUS.format(records="\n\n".join(records))


def load_longhealth_questions(
    use_cot: bool,
) -> list[tuple[str, list[str], str]]:
    # 문제마다 (question, 선택지, 정답), use_cot이면 cot을 먼저 쓰라는 question
    template = LONGHEALTH_QUESTION if use_cot else LONGHEALTH_NO_COT_QUESTION
    problems = []
    for patient_id, patient in load_longhealth_patients():
        for row in patient["questions"]:
            options = [row[f"answer_{letter}"] for letter in OPTION_LETTERS]
            question = template.format(patient_id=patient_id, question=row["question"], options="\n".join(options), **patient)
            problems.append((question, options, row["correct"].strip().lower()))
    return problems
