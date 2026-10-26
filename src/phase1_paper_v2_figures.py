"""Build Korean v2 figures and detail tables from canonical result artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/paper-v2-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/phase1/canonical"
OUT = ROOT / "paper/manuscripts/v2"
FIG = OUT / "figures"
TABLE = OUT / "tables"
CLASSES = ["ordinario", "pizzicato_secco", "tremolo", "sul_ponticello", "col_legno_battuto"]
SHORT = ["ord.", "pizz.", "trem.", "sul pont.", "col legno"]
INSTRUMENTS = ["Violin", "Viola", "Violoncello", "Contrabass"]
KOREAN = ["바이올린", "비올라", "첼로", "콘트라베이스"]
RATIOS = [0.05, 0.1, 0.2, 0.5]
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#666666"
INPUTS = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative):
    path = BASE / relative
    INPUTS[str(path.relative_to(ROOT))] = digest(path)
    return pd.read_csv(path)


def save(fig, name):
    for extension in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def matrix(frame):
    assert len(frame) == 25
    counts = frame.pivot(index="true_class", columns="predicted_class", values="count").loc[CLASSES, CLASSES].to_numpy()
    assert counts.sum() == 3402 * 3 and (counts >= 0).all()
    return counts / counts.sum(axis=1, keepdims=True)


def protocol():
    fig, ax = plt.subplots(figsize=(10.4, 6.2), layout="constrained")
    ax.set(xlim=(0, 10), ylim=(0, 7))
    ax.axis("off")

    def box(x, y, w, h, label, color="#EAF2F8"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07", facecolor=color, edgecolor="#506070", lw=1))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=11.5, linespacing=1.5)

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, color="#506070", lw=1.2))

    box(0.3, 5.8, 5.8, 0.8, "외부 학습: 세 악기종\n이 안에서 라벨 예산 5 / 10 / 20 / 50% 설정")
    box(7.1, 5.8, 2.5, 0.8, "외부 테스트: 한 악기종\n학습·설정 선택에 사용하지 않음", "#F8ECE7")
    box(0.3, 4.5, 3.4, 0.7, "관측 라벨 시드\n예산마다 세 번 표집")
    box(4.2, 4.5, 1.9, 0.7, "미라벨 풀\n자기학습에서 사용", "#F4F4F4")
    arrow((1.8, 5.8), (1.8, 5.2)); arrow((5.1, 5.8), (5.1, 5.2))
    box(0.15, 2.9, 1.85, 0.85, "학습\n시드의 약 64%")
    box(2.25, 2.9, 1.85, 0.85, "조기종료 검증\n시드의 약 16%")
    box(4.35, 2.9, 1.85, 0.85, "설정·반복 선택\n시드의 약 20%")
    for x in (1.05, 3.15, 5.25):
        arrow((2.0, 4.5), (x, 3.75))
    box(0.15, 1.25, 3.95, 0.85, "학습 데이터에서 scaler·class weight 계산\n조기종료 검증으로 학습 상태 선택")
    box(4.35, 1.25, 2.05, 0.85, "검증으로 설정 선택\n선택 checkpoint 저장")
    box(7.1, 1.25, 2.5, 0.85, "checkpoint 재로딩\n테스트 예측 → 지표·그림", "#F8ECE7")
    arrow((1.05, 2.9), (1.05, 2.1)); arrow((3.15, 2.9), (3.15, 2.1))
    arrow((5.25, 2.9), (5.25, 2.1)); arrow((4.1, 1.68), (4.35, 1.68))
    arrow((6.4, 1.68), (7.1, 1.68)); arrow((8.35, 5.8), (8.35, 2.1))
    ax.text(0.2, 0.55, "자기학습: 미라벨 표본을 채택할 때 (표본, 예측 target)을 함께 보존한다.\n세 라벨 분할은 서로 겹치지 않으며, 반올림 때문에 비율은 근사값이다.", fontsize=11.5, va="center")
    save(fig, "figure2_protocol")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": ["AppleGothic", "Arial Unicode MS", "DejaVu Sans"],
                         "font.size": 12, "axes.unicode_minus": False, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    meta = read("release/canonical-reproduction/data/release_meta.csv")
    model = read("mert_svm/canonical_summary.csv").set_index("seed_ratio").loc[RATIOS]
    runs = read("mert_svm/canonical_results.csv")
    st = read("self_training/canonical_summary.csv")
    per_class = read("verification/per_class_f1.csv")
    intervals = read("verification/bootstrap_sensitivity.csv")
    counts = read("verification/confusion_counts.csv")
    st_counts = read("self_training/confusion_counts.csv")
    read("self_training/per_class_f1.csv").to_csv(TABLE / "self_training_per_class.csv", index=False)
    read("cnn/canonical_results.csv").to_csv(TABLE / "cnn_reference.csv", index=False)
    data_counts = pd.crosstab(meta.instrument, meta.label).loc[INSTRUMENTS, CLASSES]
    assert data_counts.to_numpy().sum() == 3402
    data_counts.to_csv(TABLE / "dataset_counts.csv")
    model.to_csv(TABLE / "model_summary.csv")
    st.to_csv(TABLE / "self_training_summary.csv", index=False)
    per_class.to_csv(TABLE / "per_class_f1.csv", index=False)
    intervals.to_csv(TABLE / "bootstrap_sensitivity.csv", index=False)
    folds = runs.groupby(["seed_ratio", "held_out_instrument"])[["mert_macro_f1", "svm_macro_f1", "mert_minus_svm"]].mean().reset_index()
    folds.to_csv(TABLE / "instrument_means.csv", index=False)
    partitions = []
    for path in sorted((BASE / "mert_svm/manifests").glob("*.json")):
        INPUTS[str(path.relative_to(ROOT))] = digest(path)
        m = json.loads(path.read_text())
        partitions.append({"ratio": m["seed_ratio"], "split_key": m["split_key"], **m["partition_sizes"]})
    pd.DataFrame(partitions).to_csv(TABLE / "partition_sizes.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.8, 3.5), layout="constrained")
    values = data_counts.to_numpy()
    im = ax.imshow(values, cmap="Blues", vmin=0, vmax=values.max(), aspect="auto")
    ax.set_xticks(range(5), SHORT); ax.set_yticks(range(4), KOREAN)
    ax.set(xlabel="주법", ylabel="악기종")
    for r in range(4):
        for c in range(5):
            ax.text(c, r, str(values[r, c]), ha="center", va="center", color="white" if values[r,c] > values.max()*.6 else "black")
    fig.colorbar(im, ax=ax, label="고유 클립 수")
    save(fig, "figure1_dataset")
    protocol()

    x = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
    for name, label, color, marker in [("mert", "MERT 헤드", BLUE, "o"), ("svm", "특징 SVM", ORANGE, "s")]:
        axes[0].plot(x, model[f"{name}_macro_f1_fold_mean"], marker=marker, color=color, label=label)
    for name, label, color, style in [("supervised", "감독 대조군", BLUE, "o-"), ("pseudo", "의사 라벨", ORANGE, "s-"), ("oracle", "Oracle 진단", GREY, "^--")]:
        rows = st[st.condition == name].set_index("ratio").loc[RATIOS]
        axes[1].plot(x, rows.macro_f1_fold_mean, style, color=color, label=label)
    for ax, title in zip(axes, ["(a) 검증으로 튜닝한 모델", "(b) 자기학습 내부 비교"]):
        ax.set(title=title, xlabel="관측 라벨 예산", ylabel="악기종 평균 macro F1", ylim=(0.60, 0.91))
        ax.set_xticks(x, ["5%", "10%", "20%", "50%"])
        ax.legend(loc="lower right", fontsize=11.5); ax.grid(axis="y", alpha=.2)
    save(fig, "figure3_budget_curves")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), layout="constrained")
    colors = [BLUE, ORANGE, "#009E73", "#8B5BA7"]
    for ins, label, color, marker in zip(INSTRUMENTS, KOREAN, colors, ["o", "s", "^", "D"]):
        rows = folds[folds.held_out_instrument == ins].set_index("seed_ratio").loc[RATIOS]
        axes[0].plot(x, rows.mert_minus_svm, marker=marker, label=label, color=color)
    axes[0].axhline(0, color=GREY, ls="--", lw=1)
    axes[0].set_xticks(x, ["5%", "10%", "20%", "50%"])
    axes[0].set(title="(a) 보류 악기종별 차이", ylabel="MERT − SVM macro F1", xlabel="관측 라벨 예산", ylim=(-.11,.11))
    axes[0].legend(fontsize=11.5, ncols=2, loc="lower right"); axes[0].grid(axis="y", alpha=.2)
    for name, offset, label, color, marker in [("mert", -.08, "MERT 헤드", BLUE, "o"), ("svm", .08, "특징 SVM", ORANGE, "s")]:
        rows = per_class[(per_class.condition=="ratio_0.5") & (per_class.model==name)].set_index("class").loc[CLASSES]
        axes[1].plot(rows.fold_mean_f1, np.arange(5)+offset, marker, color=color, label=label)
    axes[1].set_yticks(range(5), SHORT); axes[1].invert_yaxis()
    axes[1].set(title="(b) 주법별 F1: 라벨 50%", xlabel="악기종 평균 클래스 F1", xlim=(.65,1.01))
    axes[1].grid(axis="x", alpha=.2); axes[1].legend(fontsize=11.5, loc="lower right")
    save(fig, "figure4_detailed_performance")

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.2), layout="constrained")
    panels = [(counts[(counts.condition=="ratio_0.5") & (counts.model=="mert")], "(a) MERT 헤드: 50%"),
              (counts[(counts.condition=="ratio_0.5") & (counts.model=="svm")], "(b) 특징 SVM: 50%"),
              (st_counts[(st_counts.ratio==.05) & (st_counts.condition=="supervised")], "(c) 감독 대조군: 5%"),
              (st_counts[(st_counts.ratio==.05) & (st_counts.condition=="pseudo")], "(d) 의사 라벨: 5%")]
    for ax, (frame, title) in zip(axes.flat, panels):
        values = matrix(frame)
        im = ax.imshow(values, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(5), SHORT, rotation=30, ha="right"); ax.set_yticks(range(5), SHORT)
        ax.set(title=title, xlabel="예측 주법", ylabel="실제 주법")
        for r in range(5):
            for c in range(5):
                ax.text(c,r,f"{values[r,c]:.2f}",ha="center",va="center",fontsize=11.5,color="white" if values[r,c]>.55 else "black")
    fig.colorbar(im, ax=axes, shrink=.75, label="합산 예측의 행 비율")
    save(fig, "figure5_confusions")

    fig, ax = plt.subplots(figsize=(8.8,4),layout="constrained")
    for offset, unit, label, color, marker in [(-.18,"clip","클립",BLUE,"o"),(0,"recording_proxy","파일명 proxy","#009E73","s"),(.18,"instrument_type","악기종",ORANGE,"^")]:
        rows=intervals[(intervals.unit==unit)&(intervals.measure=="mert_minus_svm")].set_index("seed_ratio").loc[RATIOS]
        assert (rows.draws==1000).all() and np.allclose(rows.point,model.mert_minus_svm)
        ax.errorbar(x+offset,rows.point,yerr=np.vstack([rows.point-rows.ci_low,rows.ci_high-rows.point]),fmt=marker,capsize=4,color=color,label=label)
    ax.axhline(0,color=GREY,ls="--",lw=1)
    ax.set_xticks(x,["5%","10%","20%","50%"])
    ax.set(xlabel="관측 라벨 예산",ylabel="MERT − SVM macro F1",ylim=(-.075,.09))
    ax.legend(ncols=3,fontsize=11.5,loc="upper right"); ax.grid(axis="y",alpha=.2)
    save(fig,"figure6_resampling")

    # Protocol artwork is derived from these implementations, not measured data.
    for name in ["phase1_canonical_mert_svm.py", "phase1_selftraining_repaired.py", "phase1_canonical_selftraining.py"]:
        path=ROOT / "src" / name
        INPUTS[str(path.relative_to(ROOT))]=digest(path)
    manifest={"script":str(Path(__file__).relative_to(ROOT)),"script_sha256":digest(Path(__file__)),
              "inputs":INPUTS,"outputs":{str(p.relative_to(OUT)):digest(p) for folder in [FIG,TABLE] for p in sorted(folder.iterdir()) if p.is_file() and p.name!="manifest.json"},
              "figures":6,"dpi":300,"font":"AppleGothic; fallback Arial Unicode MS",
              "aggregation":"mean three trials within instrument type; equal mean four types",
              "confusion":"row normalized pooled counts; three predictions per unique clip",
              "intervals":"1000 paired percentile resamples on fixed predictions; no equivalence test",
              "protocol":"schematic; approximately 64/16/20 percent of observed seed; no empirical effect size"}
    (FIG / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(f"Built six figures (PNG/PDF) and {len(list(TABLE.glob('*.csv')))} detail tables.")


if __name__ == "__main__":
    main()
