"""Generate report-ready plots from LMLM audit metrics CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_MPL_CONFIG_DIR = Path("/tmp/lmlm-audit-matplotlib")
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src" / "lmlm-audit"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics import metrics_total


WANDB_PROJECT = "lmlm-audit-analysis"

STATE_ORDER = ["FULL", "DEL-ON", "DEL-OFF"]
STATE_LABELS = {
    "FULL": "FULL",
    "DEL-ON": "DEL-ON",
    "DEL-OFF": "DEL-OFF",
}
STATE_COLORS = {
    "FULL": "#2563eb",
    "DEL-ON": "#16a34a",
    "DEL-OFF": "#dc2626",
}
VARIANT_ORDER = ["base", "alias", "collision", "noise"]
VARIANT_LABELS = {
    "base": "Base",
    "alias": "Alias",
    "collision": "Collision",
    "noise": "Noise",
    "released_lmlm": "Released LMLM",
}
PROMPT_ORDER = [
    "direct_questions",
    "contextual_questions",
    "paraphrased_questions",
    "cloze",
    "continuations",
    "fewshot_questions",
]
PROMPT_LABELS = {
    "direct_questions": "Direct",
    "contextual_questions": "Contextual",
    "paraphrased_questions": "Paraphrased",
    "cloze": "Cloze",
    "continuations": "Continuations",
    "fewshot_questions": "Few-shot",
}
METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "Token F1",
    "parametric_leakage": "Parametric leakage",
    "retrieval_artifact_rate": "Retrieval artifact rate",
}
DECOMPOSITION_COLORS = {
    "parametric_leakage": "#dc2626",
    "retrieval_artifact_rate": "#f59e0b",
}
CUSTOM_DOMAINS = {"countries", "politicians", "sports"}
CUSTOM_VARIANTS = {"base", "alias", "collision", "noise"}
RELEASED_DOMAIN = "released_database"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_released_lmlm_metrics(paths: list[Path]) -> dict[str, float] | None:
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        return None

    results: list[dict[str, object]] = []
    for path in existing_paths:
        results.extend(read_jsonl(path))
    if not results:
        return None
    return metrics_total(results)


def add_derived_columns(row: dict[str, str]) -> dict[str, str]:
    prompt_path = Path(row["prompt_file"])
    database_path = Path(row["database_path"])
    row = dict(row)
    row["domain"] = database_path.parent.name
    row["variant"] = database_path.stem
    row["prompt_type"] = prompt_path.stem.removeprefix("prompts_")
    return row


def is_custom_row(row: dict[str, str]) -> bool:
    return row["domain"] in CUSTOM_DOMAINS and row["variant"] in CUSTOM_VARIANTS


def is_released_row(row: dict[str, str]) -> bool:
    return row["domain"] == RELEASED_DOMAIN


def filter_custom(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if is_custom_row(row)]


def filter_released(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if is_released_row(row)]


def numeric(row: dict[str, str], column: str) -> float:
    return float(row[column])


def weight(row: dict[str, str]) -> float:
    return numeric(row, "count")


def weighted_average(rows: list[dict[str, str]], metric: str) -> float:
    total_weight = sum(weight(row) for row in rows)
    if total_weight == 0:
        return 0.0
    return sum(numeric(row, metric) * weight(row) for row in rows) / total_weight


def released_metrics_from_cross_state(
    cross_state_rows: list[dict[str, str]],
) -> dict[str, float] | None:
    released_rows = filter_released(cross_state_rows)
    if not released_rows:
        return None
    metric_cols = [
        "precision",
        "recall",
        "f1",
        "parametric_leakage",
        "retrieval_mediated_correctness",
        "retrieval_artifact_rate",
    ]
    return {metric: weighted_average(released_rows, metric) for metric in metric_cols}


def released_per_state_metrics(
    per_state_rows: list[dict[str, str]],
) -> dict[str, dict[str, float]] | None:
    released_rows = filter_released(per_state_rows)
    if not released_rows:
        return None
    metric_cols = ["exact_match", "precision", "recall", "f1"]
    grouped = group_rows(released_rows, ("state",))
    return {
        key[0]: {metric: weighted_average(rows, metric) for metric in metric_cols}
        for key, rows in grouped.items()
    }


def group_rows(
    rows: list[dict[str, str]],
    keys: tuple[str, ...],
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return dict(grouped)


def ordered_values(values: set[str], preferred_order: list[str]) -> list[str]:
    known = [value for value in preferred_order if value in values]
    extra = sorted(value for value in values if value not in preferred_order)
    return known + extra


def pretty_prompt(prompt_type: str) -> str:
    return PROMPT_LABELS.get(prompt_type, prompt_type.replace("_", " ").title())


def pretty_variant(variant: str) -> str:
    return VARIANT_LABELS.get(variant, variant.replace("_", " ").title())


def percent_axis(ax: plt.Axes, upper: float = 1.0) -> None:
    ax.set_ylim(0, upper)
    ticks = np.linspace(0, upper, 6)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{int(round(value * 100))}%" for value in ticks])
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)


def save_figure(
    fig: plt.Figure, output_dir: Path, stem: str, tight: bool = True
) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cross_state_diagnostic_by_variant(
    cross_state_rows: list[dict[str, str]],
    output_dir: Path,
    metric: str,
    color: str,
    stem: str,
    released_lmlm_metrics: dict[str, float] | None = None,
) -> None:
    variants = ordered_values(
        {row["variant"] for row in cross_state_rows}, VARIANT_ORDER
    )
    grouped = group_rows(cross_state_rows, ("variant",))
    labels = [pretty_variant(variant) for variant in variants]
    values = [
        weighted_average(grouped.get((variant,), []), metric) for variant in variants
    ]

    if released_lmlm_metrics is not None:
        labels.append(pretty_variant("released_lmlm"))
        values.append(released_lmlm_metrics[metric])

    max_value = max(values) if values else 0.0
    upper = max(0.02, min(1.0, np.ceil((max_value + 0.01) * 20) / 20))

    x = np.arange(len(labels))
    colors = [color] * len(labels)
    if released_lmlm_metrics is not None:
        colors[-1] = "#64748b"

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    bars = ax.bar(x, values, color=colors, width=0.62)
    ax.set_title(f"Average {METRIC_LABELS[metric]} across databases")
    ax.set_ylabel("Weighted rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    percent_axis(ax, upper=upper)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + upper * 0.02,
            f"{value * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save_figure(fig, output_dir, stem)


def default_released_result_paths(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("prompts_*_results.jsonl"))


def plot_exact_match_by_variant_state(
    per_state_rows: list[dict[str, str]],
    output_dir: Path,
    released_per_state: dict[str, dict[str, float]] | None = None,
) -> None:
    variants = ordered_values({row["variant"] for row in per_state_rows}, VARIANT_ORDER)
    grouped = group_rows(per_state_rows, ("variant", "state"))

    labels = [pretty_variant(variant) for variant in variants]
    if released_per_state is not None:
        labels.append(pretty_variant("released_lmlm"))

    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    for idx, state in enumerate(STATE_ORDER):
        values = [
            weighted_average(grouped.get((variant, state), []), "exact_match")
            for variant in variants
        ]
        if released_per_state is not None:
            values.append(released_per_state.get(state, {}).get("exact_match", 0.0))
        ax.bar(
            x + (idx - 1) * width,
            values,
            width,
            color=STATE_COLORS[state],
            label=STATE_LABELS[state],
        )

    ax.set_title("Exact match by variant and intervention state")
    ax.set_ylabel("Weighted exact match")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    percent_axis(ax)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, output_dir, "exact_match_by_variant_state")


def plot_del_on_attribution_by_variant(
    cross_state_rows: list[dict[str, str]],
    output_dir: Path,
    released_lmlm_metrics: dict[str, float] | None = None,
) -> None:
    variants = ordered_values(
        {row["variant"] for row in cross_state_rows}, VARIANT_ORDER
    )
    grouped = group_rows(cross_state_rows, ("variant",))

    leakage = [
        weighted_average(grouped.get((variant,), []), "parametric_leakage")
        for variant in variants
    ]
    rmc = [
        weighted_average(grouped.get((variant,), []), "retrieval_mediated_correctness")
        for variant in variants
    ]
    artifact = [
        weighted_average(grouped.get((variant,), []), "retrieval_artifact_rate")
        for variant in variants
    ]
    labels = [pretty_variant(variant) for variant in variants]

    if released_lmlm_metrics is not None:
        labels.append(pretty_variant("released_lmlm"))
        leakage.append(released_lmlm_metrics.get("parametric_leakage", 0.0))
        rmc.append(released_lmlm_metrics.get("retrieval_mediated_correctness", 0.0))
        artifact.append(released_lmlm_metrics.get("retrieval_artifact_rate", 0.0))

    leakage_arr = np.array(leakage)
    rmc_arr = np.array(rmc)
    artifact_arr = np.array(artifact)

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(
        x - width / 2,
        leakage_arr,
        width,
        color=DECOMPOSITION_COLORS["parametric_leakage"],
        label=r"Parametric leakage $L(f)$ ($\hat{L}$)",
    )
    ax.bar(
        x - width / 2,
        rmc_arr,
        width,
        bottom=leakage_arr,
        color="#16a34a",
        label=r"Retrieval-mediated correctness $R(f)$",
    )
    ax.bar(
        x + width / 2,
        artifact_arr,
        width,
        color=DECOMPOSITION_COLORS["retrieval_artifact_rate"],
        label="Retrieval artifact rate",
    )

    stacked = leakage_arr + rmc_arr
    for i, total in enumerate(stacked):
        if total > 0:
            ax.text(
                x[i] - width / 2,
                total + 0.005,
                f"{total * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    stacked_max = float(stacked.max(initial=0.0))
    artifact_max = float(artifact_arr.max(initial=0.0))
    upper = max(
        0.05, min(1.0, np.ceil((max(stacked_max, artifact_max) + 0.02) * 20) / 20)
    )

    ax.set_title("Attribution of DEL-ON correctness by variant")
    ax.set_ylabel("Weighted rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    percent_axis(ax, upper=upper)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, output_dir, "del_on_correctness_attribution_by_variant")


def plot_del_on_attribution_by_prompt(
    cross_state_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    prompts = ordered_values(
        {row["prompt_type"] for row in cross_state_rows}, PROMPT_ORDER
    )
    grouped = group_rows(cross_state_rows, ("prompt_type",))

    leakage = [
        weighted_average(grouped.get((prompt,), []), "parametric_leakage")
        for prompt in prompts
    ]
    rmc = [
        weighted_average(grouped.get((prompt,), []), "retrieval_mediated_correctness")
        for prompt in prompts
    ]
    artifact = [
        weighted_average(grouped.get((prompt,), []), "retrieval_artifact_rate")
        for prompt in prompts
    ]
    labels = [pretty_prompt(prompt) for prompt in prompts]

    leakage_arr = np.array(leakage)
    rmc_arr = np.array(rmc)
    artifact_arr = np.array(artifact)

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(
        x - width / 2,
        leakage_arr,
        width,
        color=DECOMPOSITION_COLORS["parametric_leakage"],
        label=r"Parametric leakage $L(f)$ ($\hat{L}$)",
    )
    ax.bar(
        x - width / 2,
        rmc_arr,
        width,
        bottom=leakage_arr,
        color="#16a34a",
        label=r"Retrieval-mediated correctness $R(f)$",
    )
    ax.bar(
        x + width / 2,
        artifact_arr,
        width,
        color=DECOMPOSITION_COLORS["retrieval_artifact_rate"],
        label="Retrieval artifact rate",
    )

    stacked = leakage_arr + rmc_arr
    for i, total in enumerate(stacked):
        if total > 0:
            ax.text(
                x[i] - width / 2,
                total + 0.005,
                f"{total * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    stacked_max = float(stacked.max(initial=0.0))
    artifact_max = float(artifact_arr.max(initial=0.0))
    upper = max(
        0.05, min(1.0, np.ceil((max(stacked_max, artifact_max) + 0.02) * 20) / 20)
    )

    ax.set_title("Attribution of DEL-ON correctness by prompt style")
    ax.set_ylabel("Weighted rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    percent_axis(ax, upper=upper)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, output_dir, "del_on_correctness_attribution_by_prompt")


def plot_overlap_metrics_by_prompt_state(
    per_state_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    prompts = ordered_values(
        {row["prompt_type"] for row in per_state_rows},
        PROMPT_ORDER,
    )
    metrics = ["precision", "recall", "f1"]
    grouped = group_rows(per_state_rows, ("prompt_type", "state"))
    x = np.arange(len(prompts))
    width = 0.24

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(16, 5.2),
        sharey=True,
        constrained_layout=False,
    )
    for ax, metric in zip(axes, metrics, strict=True):
        for idx, state in enumerate(STATE_ORDER):
            values = [
                weighted_average(grouped.get((prompt, state), []), metric)
                for prompt in prompts
            ]
            ax.bar(
                x + (idx - 1) * width,
                values,
                width,
                color=STATE_COLORS[state],
                label=STATE_LABELS[state],
            )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(
            [pretty_prompt(prompt) for prompt in prompts], rotation=30, ha="right"
        )
        percent_axis(ax)

    axes[0].set_ylabel("Weighted rate")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.subplots_adjust(top=0.76, bottom=0.24, wspace=0.12)
    fig.suptitle(
        "Average precision, recall, and token F1 by prompt style and state", y=0.98
    )
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
    )
    save_figure(fig, output_dir, "precision_recall_f1_by_prompt_and_state", tight=False)


def plot_token_f1_by_prompt_state(
    per_state_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    prompts = ordered_values(
        {row["prompt_type"] for row in per_state_rows},
        PROMPT_ORDER,
    )
    grouped = group_rows(per_state_rows, ("prompt_type", "state"))
    x = np.arange(len(prompts))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    for idx, state in enumerate(STATE_ORDER):
        values = [
            weighted_average(grouped.get((prompt, state), []), "f1")
            for prompt in prompts
        ]
        ax.bar(
            x + (idx - 1) * width,
            values,
            width,
            color=STATE_COLORS[state],
            label=STATE_LABELS[state],
        )
    ax.set_title("Token F1 by prompt style and state")
    ax.set_ylabel("Weighted rate")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [pretty_prompt(prompt) for prompt in prompts], rotation=30, ha="right"
    )
    percent_axis(ax)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, output_dir, "token_f1_by_prompt_and_state")


def init_wandb_run(project: str) -> Any:
    from dotenv import load_dotenv

    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path, override=True)

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError(f"WANDB_API_KEY was not found after loading {env_path}.")

    import wandb

    wandb.login(key=api_key, relogin=True)
    return wandb.init(project=project, name="audit_plots", reinit="finish_previous")


def log_plots_to_wandb(run: Any, output_dir: Path) -> None:
    import wandb

    images = {
        path.stem: wandb.Image(str(path)) for path in sorted(output_dir.glob("*.png"))
    }
    if images:
        run.log(images)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-state",
        type=Path,
        default=Path("outputs/audit/per_state_metrics.csv"),
        help="Path to per-state metrics CSV.",
    )
    parser.add_argument(
        "--cross-state",
        type=Path,
        default=Path("outputs/audit/cross_state_metrics.csv"),
        help="Path to cross-state metrics CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/audit/plots"),
        help="Directory for generated plots.",
    )
    parser.add_argument(
        "--released-result-files",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Optional raw result JSONL files generated from "
            "`data/released_database/lmlm_database.json` and its prompts. "
            "Defaults to `outputs/audit/prompts_*_results.jsonl`."
        ),
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help=f"Log generated plots to the `{WANDB_PROJECT}` wandb project.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_per_state_rows = [add_derived_columns(row) for row in read_csv(args.per_state)]
    raw_cross_state_rows = [
        add_derived_columns(row) for row in read_csv(args.cross_state)
    ]
    per_state_rows = filter_custom(raw_per_state_rows)
    cross_state_rows = filter_custom(raw_cross_state_rows)

    released_result_paths = (
        args.released_result_files
        if args.released_result_files is not None
        else default_released_result_paths(args.cross_state.parent)
    )
    released_lmlm_metrics = read_released_lmlm_metrics(released_result_paths)
    if released_lmlm_metrics is None:
        released_lmlm_metrics = released_metrics_from_cross_state(raw_cross_state_rows)
    released_per_state = released_per_state_metrics(raw_per_state_rows)

    plot_exact_match_by_variant_state(
        per_state_rows,
        args.output_dir,
        released_per_state=released_per_state,
    )
    plot_del_on_attribution_by_variant(
        cross_state_rows,
        args.output_dir,
        released_lmlm_metrics=released_lmlm_metrics,
    )
    plot_del_on_attribution_by_prompt(cross_state_rows, args.output_dir)
    plot_cross_state_diagnostic_by_variant(
        cross_state_rows,
        args.output_dir,
        metric="parametric_leakage",
        color=DECOMPOSITION_COLORS["parametric_leakage"],
        stem="average_parametric_leakage_by_domain_variant",
        released_lmlm_metrics=released_lmlm_metrics,
    )
    plot_cross_state_diagnostic_by_variant(
        cross_state_rows,
        args.output_dir,
        metric="retrieval_artifact_rate",
        color=DECOMPOSITION_COLORS["retrieval_artifact_rate"],
        stem="average_retrieval_artifact_rate_by_domain_variant",
        released_lmlm_metrics=released_lmlm_metrics,
    )
    plot_overlap_metrics_by_prompt_state(per_state_rows, args.output_dir)
    plot_token_f1_by_prompt_state(per_state_rows, args.output_dir)

    print(f"Wrote plots to {args.output_dir}")

    if args.wandb:
        run = init_wandb_run(WANDB_PROJECT)
        try:
            log_plots_to_wandb(run, args.output_dir)
        finally:
            run.finish()
        print(f"Logged plots to wandb project '{WANDB_PROJECT}'.")


if __name__ == "__main__":
    main()
