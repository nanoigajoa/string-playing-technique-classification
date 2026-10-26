"""Generate English v2 figures from the Korean generator's numerical inputs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import phase1_paper_v2_figures as source
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper/manuscripts/v2"
TRANSLATIONS = {
    "바이올린": "Violin", "비올라": "Viola", "첼로": "Cello", "콘트라베이스": "Double bass",
    "주법": "Technique", "악기종": "Instrument type", "고유 클립 수": "Unique clips",
    "MERT 헤드": "MERT head", "특징 SVM": "Feature SVM", "감독 대조군": "Supervised control",
    "의사 라벨": "Pseudo-label", "Oracle 진단": "Oracle diagnostic", "관측 라벨 예산": "Observed-label budget",
    "악기종 평균 macro F1": "Instrument-type mean macro F1",
    "악기종 평균 클래스 F1": "Instrument-type mean class F1",
    "(a) 검증으로 튜닝한 모델": "(a) Validation-tuned classifiers",
    "(b) 자기학습 내부 비교": "(b) Matched self-training comparison",
    "(a) 보류 악기종별 차이": "(a) Difference by held-out instrument type",
    "(b) 주법별 F1: 라벨 50%": "(b) Class F1: 50% labels",
    "(a) MERT 헤드: 50%": "(a) MERT head: 50%",
    "(b) 특징 SVM: 50%": "(b) Feature SVM: 50%",
    "(c) 감독 대조군: 5%": "(c) Supervised control: 5%",
    "(d) 의사 라벨: 5%": "(d) Pseudo-label: 5%",
    "예측 주법": "Predicted technique", "실제 주법": "True technique",
    "합산 예측의 행 비율": "Row fraction of pooled predictions",
    "클립": "Clip", "파일명 proxy": "Filename proxy",
    "외부 학습: 세 악기종\n이 안에서 라벨 예산 5 / 10 / 20 / 50% 설정":
        "Outer training: three instrument types\nObserved labels: 5 / 10 / 20 / 50%",
    "외부 테스트: 한 악기종\n학습·설정 선택에 사용하지 않음":
        "Outer test: one instrument type\nExcluded from training\nand configuration selection",
    "관측 라벨 시드\n예산마다 세 번 표집": "Observed-label seed\nThree samplings per budget",
    "미라벨 풀\n자기학습에서 사용": "Unlabeled pool\nUsed in self-training",
    "학습\n시드의 약 64%": "Direct training\n~64% of seed",
    "조기종료 검증\n시드의 약 16%": "Early-stopping\nvalidation\n~16% of seed",
    "설정·반복 선택\n시드의 약 20%": "Configuration /\niteration selection\n~20% of seed",
    "학습 데이터에서 scaler·class weight 계산\n조기종료 검증으로 학습 상태 선택":
        "Fit scaler and class weights on training data\nSelect state using early-stopping validation",
    "검증으로 설정 선택\n선택 checkpoint 저장":
        "Select by validation\nSave selected\ncheckpoint",
    "checkpoint 재로딩\n테스트 예측 → 지표·그림":
        "Reload checkpoint\nTest predictions\nMetrics and figures",
    "자기학습: 미라벨 표본을 채택할 때 (표본, 예측 target)을 함께 보존한다.\n세 라벨 분할은 서로 겹치지 않으며, 반올림 때문에 비율은 근사값이다.":
        "Self-training retains each accepted (example, predicted target) pair.\nLabel partitions are disjoint; percentages are approximate because of rounding.",
}


def main():
    original_save = source.save
    original_generator = Path(source.__file__)
    source.FIG = OUT / "figures_en"
    source.TABLE = OUT / "tables_en"
    source.KOREAN = ["Violin", "Viola", "Cello", "Double bass"]
    source.INPUTS[str(original_generator.relative_to(ROOT))] = source.digest(original_generator)
    source.__file__ = str(Path(__file__).resolve())

    def save_english(fig, name):
        for item in fig.findobj(match=Text):
            value = item.get_text()
            if re.search(r"[가-힣]", value):
                assert value in TRANSLATIONS, value
                item.set_text(TRANSLATIONS[value])
            item.set_fontfamily("DejaVu Sans")
        for ax in fig.axes:
            ax.title.set_fontsize(11)
            ax.xaxis.label.set_fontsize(11.5)
            ax.yaxis.label.set_fontsize(11.5)
            legend = ax.get_legend()
            if legend:
                for item in legend.get_texts():
                    item.set_fontsize(11)
        if name == "figure2_protocol":
            fig.set_size_inches(11.2, 6.5)
            for item in fig.findobj(match=Text):
                item.set_fontsize(12.5)
        if name == "figure5_confusions":
            for ax in fig.axes:
                ax.tick_params(labelsize=10)
        original_save(fig, name)

    source.save = save_english
    source.main()
    path = source.FIG / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update({"language": "en", "font": "DejaVu Sans",
                     "source_generator": str(original_generator.relative_to(ROOT)),
                     "translation_map": TRANSLATIONS,
                     "korean_figures_modified": False})
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
