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
from matplotlib.lines import Line2D

from metrics import metrics_total


WANDB_PROJECT = "lmlm-audit-analysis"

STATE_ORDER = ["FULL", "DEL-ON", "DEL-OFF"]
STATE_LABELS = {
    "FULL": "Full DB",
    "DEL-ON": "Deleted + retrieval",
    "DEL-OFF": "Deleted, no retrieval",
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
DOMAIN_ORDER = ["countries", "politicians", "sports"]
DOMAIN_LABELS = {
    "countries": "Countries",
    "politicians": "Politicians",
    "sports": "Sports",
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
    "direct_questions": "Direct Qs",
    "contextual_questions": "Contextual Qs",
    "paraphrased_questions": "Paraphrased Qs",
    "cloze": "Cloze",
    "continuations": "Continuations",
    "fewshot_questions": "Few-shot Qs",
}
METRIC_LABELS = {
    "exact_match": "Exact match",
    "contains_match": "Contains match",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "Token F1",
    "unknown_rate": "Unknown rate",
    "parametric_leakage": "Parametric leakage",
    "retrieval_mediated_correctness": "Retrieval-mediated correctness",
    "retrieval_artifact_rate": "Retrieval artifact rate",
}
DECOMPOSITION_METRICS = [
    "parametric_leakage",
    "retrieval_mediated_correctness",
    "retrieval_artifact_rate",
]
DECOMPOSITION_COLORS = {
    "parametric_leakage": "#dc2626",
    "retrieval_mediated_correctness": "#16a34a",
    "retrieval_artifact_rate": "#f59e0b",
}
VARIANT_COLORS = {
    "base": "#2563eb",
    "alias": "#16a34a",
    "collision": "#dc2626",
    "noise": "#f59e0b",
}
DOMAIN_MARKERS = {
    "countries": "o",
    "politicians": "s",
    "sports": "D",
}


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


def numeric(row: dict[str, str], column: str) -> float:
    return float(row[column])


def weight(row: dict[str, str]) -> float:
    return numeric(row, "count")


def weighted_average(rows: list[dict[str, str]], metric: str) -> float:
    total_weight = sum(weight(row) for row in rows)
    if total_weight == 0:
        return 0.0
    return sum(numeric(row, metric) * weight(row) for row in rows) / total_weight


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


def pretty_domain(domain: str) -> str:
    return DOMAIN_LABELS.get(domain, domain.replace("_", " ").title())


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


def plot_overall_metrics(
    per_state_rows: list[dict[str, str]], output_dir: Path
) -> None:
    metrics = ["exact_match", "contains_match", "f1", "unknown_rate"]
    grouped = group_rows(per_state_rows, ("state",))
    x = np.arange(len(metrics))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for idx, state in enumerate(STATE_ORDER):
        rows = grouped.get((state,), [])
        values = [weighted_average(rows, metric) for metric in metrics]
        ax.bar(
            x + (idx - 1) * width,
            values,
            width,
            color=STATE_COLORS[state],
            label=STATE_LABELS[state],
        )

    ax.set_title("Overall Accuracy and Abstention by Intervention State")
    ax.set_ylabel("Weighted rate")
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[metric] for metric in metrics])
    percent_axis(ax)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, output_dir, "overall_metrics_by_state")


def plot_prompt_state_accuracy(
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

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    for idx, state in enumerate(STATE_ORDER):
        values = [
            weighted_average(grouped.get((prompt, state), []), "exact_match")
            for prompt in prompts
        ]
        ax.bar(
            x + (idx - 1) * width,
            values,
            width,
            color=STATE_COLORS[state],
            label=STATE_LABELS[state],
        )

    ax.set_title("Exact Match by Prompt Style and Deletion State")
    ax.set_ylabel("Weighted exact match")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [pretty_prompt(prompt) for prompt in prompts], rotation=20, ha="right"
    )
    percent_axis(ax)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, output_dir, "exact_match_by_prompt_and_state")


def plot_domain_variant_state_accuracy(
    per_state_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    domains = ordered_values({row["domain"] for row in per_state_rows}, DOMAIN_ORDER)
    variants = ordered_values({row["variant"] for row in per_state_rows}, VARIANT_ORDER)
    grouped = group_rows(per_state_rows, ("domain", "variant", "state"))
    x = np.arange(len(variants))
    width = 0.24

    fig, axes = plt.subplots(
        1,
        len(domains),
        figsize=(15, 5),
        sharey=True,
        constrained_layout=False,
    )
    if len(domains) == 1:
        axes = [axes]

    for ax, domain in zip(axes, domains, strict=True):
        for idx, state in enumerate(STATE_ORDER):
            values = [
                weighted_average(
                    grouped.get((domain, variant, state), []), "exact_match"
                )
                for variant in variants
            ]
            ax.bar(
                x + (idx - 1) * width,
                values,
                width,
                color=STATE_COLORS[state],
                label=STATE_LABELS[state],
            )
        ax.set_title(pretty_domain(domain))
        ax.set_xticks(x)
        ax.set_xticklabels(
            [pretty_variant(variant) for variant in variants], rotation=20, ha="right"
        )
        percent_axis(ax)

    axes[0].set_ylabel("Weighted exact match")
    fig.subplots_adjust(top=0.78, bottom=0.18, wspace=0.20)
    fig.suptitle("Exact Match by Domain, Database Variant, and State", y=0.98)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
    )
    save_figure(fig, output_dir, "exact_match_by_domain_variant_state", tight=False)


def plot_deletion_drop_by_prompt(
    per_state_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    prompts = ordered_values(
        {row["prompt_type"] for row in per_state_rows},
        PROMPT_ORDER,
    )
    grouped = group_rows(per_state_rows, ("prompt_type", "state"))
    full_values = np.array(
        [
            weighted_average(grouped.get((prompt, "FULL"), []), "exact_match")
            for prompt in prompts
        ]
    )
    del_on_values = np.array(
        [
            weighted_average(grouped.get((prompt, "DEL-ON"), []), "exact_match")
            for prompt in prompts
        ]
    )
    del_off_values = np.array(
        [
            weighted_average(grouped.get((prompt, "DEL-OFF"), []), "exact_match")
            for prompt in prompts
        ]
    )

    x = np.arange(len(prompts))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.bar(
        x - width / 2,
        full_values - del_on_values,
        width,
        color="#7c3aed",
        label="Full DB to deleted + retrieval",
    )
    ax.bar(
        x + width / 2,
        full_values - del_off_values,
        width,
        color="#0891b2",
        label="Full DB to deleted, no retrieval",
    )
    ax.set_title("Accuracy Loss After Fact Deletion")
    ax.set_ylabel("Exact-match drop")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [pretty_prompt(prompt) for prompt in prompts], rotation=20, ha="right"
    )
    percent_axis(ax)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, output_dir, "accuracy_drop_after_deletion_by_prompt")


def plot_retrieval_decomposition(
    cross_state_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    domains = ordered_values({row["domain"] for row in cross_state_rows}, DOMAIN_ORDER)
    variants = ordered_values(
        {row["variant"] for row in cross_state_rows}, VARIANT_ORDER
    )
    grouped = group_rows(cross_state_rows, ("domain", "variant"))
    x_labels = [
        f"{pretty_domain(domain)}\n{pretty_variant(variant)}"
        for domain in domains
        for variant in variants
    ]
    x = np.arange(len(x_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    max_value = 0.0
    for idx, metric in enumerate(DECOMPOSITION_METRICS):
        values = np.array(
            [
                weighted_average(grouped.get((domain, variant), []), metric)
                for domain in domains
                for variant in variants
            ]
        )
        max_value = max(max_value, float(values.max(initial=0.0)))
        ax.bar(
            x + (idx - 1) * width,
            values,
            width,
            color=DECOMPOSITION_COLORS[metric],
            label=METRIC_LABELS[metric],
        )

    ax.set_title("Post-Deletion Diagnostic Rates", pad=12)
    ax.set_ylabel("Weighted rate")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    percent_axis(ax, upper=max(0.2, min(1.0, np.ceil((max_value + 0.03) * 10) / 10)))
    ax.legend(frameon=False, ncol=1, loc="upper right")
    save_figure(fig, output_dir, "retrieval_decomposition_by_domain_variant")


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

    max_value = max(values)
    upper = max(0.02, min(1.0, np.ceil((max_value + 0.01) * 20) / 20))

    x = np.arange(len(labels))
    colors = [color] * len(labels)
    if released_lmlm_metrics is not None:
        colors[-1] = "#64748b"

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    bars = ax.bar(x, values, color=colors, width=0.62)
    ax.set_title(f"Average {METRIC_LABELS[metric]} Across Databases")
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
    fig.suptitle("Average Precision, Recall, and F1 by Prompt Style and State", y=0.98)
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
    )
    save_figure(fig, output_dir, "precision_recall_f1_by_prompt_and_state", tight=False)


def annotate_heatmap(ax: plt.Axes, values: np.ndarray) -> None:
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            text_color = "white" if value >= 0.45 else "#111827"
            ax.text(
                col_idx,
                row_idx,
                f"{value * 100:.0f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )


def plot_prompt_heatmaps(
    cross_state_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    prompts = ordered_values(
        {row["prompt_type"] for row in cross_state_rows},
        PROMPT_ORDER,
    )
    variants = ordered_values(
        {row["variant"] for row in cross_state_rows}, VARIANT_ORDER
    )
    grouped = group_rows(cross_state_rows, ("variant", "prompt_type"))

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5), sharey=True, constrained_layout=True
    )
    max_value = max(
        numeric(row, metric)
        for row in cross_state_rows
        for metric in DECOMPOSITION_METRICS
    )
    vmax = max(0.2, min(1.0, np.ceil((max_value + 0.02) * 10) / 10))
    for ax, metric in zip(axes, DECOMPOSITION_METRICS, strict=True):
        values = np.array(
            [
                [
                    weighted_average(grouped.get((variant, prompt), []), metric)
                    for prompt in prompts
                ]
                for variant in variants
            ]
        )
        image = ax.imshow(values, vmin=0, vmax=vmax, cmap="YlGnBu", aspect="auto")
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(np.arange(len(prompts)))
        ax.set_xticklabels(
            [pretty_prompt(prompt) for prompt in prompts], rotation=35, ha="right"
        )
        ax.set_yticks(np.arange(len(variants)))
        ax.set_yticklabels([pretty_variant(variant) for variant in variants])
        annotate_heatmap(ax, values)

    fig.suptitle("Deletion Diagnostics by Prompt Style and Database Variant", y=1.02)
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, pad=0.01)
    colorbar.set_label("Weighted rate")
    colorbar.set_ticks(np.linspace(0, vmax, 5))
    colorbar.set_ticklabels(
        [f"{value * 100:.0f}%" for value in np.linspace(0, vmax, 5)]
    )
    save_figure(
        fig, output_dir, "deletion_diagnostics_prompt_variant_heatmaps", tight=False
    )


def wilson_ci(p: float, n: float) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z = 1.959964
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1.0 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def plot_del_on_attribution_by_prompt(
    cross_state_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    prompts = ordered_values(
        {row["prompt_type"] for row in cross_state_rows},
        PROMPT_ORDER,
    )
    grouped = group_rows(cross_state_rows, ("prompt_type",))
    leakage = np.array(
        [
            weighted_average(grouped.get((prompt,), []), "parametric_leakage")
            for prompt in prompts
        ]
    )
    rmc = np.array(
        [
            weighted_average(
                grouped.get((prompt,), []), "retrieval_mediated_correctness"
            )
            for prompt in prompts
        ]
    )
    artifact = np.array(
        [
            weighted_average(grouped.get((prompt,), []), "retrieval_artifact_rate")
            for prompt in prompts
        ]
    )

    x = np.arange(len(prompts))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(
        x - width / 2,
        leakage,
        width,
        color=DECOMPOSITION_COLORS["parametric_leakage"],
        label=METRIC_LABELS["parametric_leakage"],
    )
    ax.bar(
        x - width / 2,
        rmc,
        width,
        bottom=leakage,
        color=DECOMPOSITION_COLORS["retrieval_mediated_correctness"],
        label=METRIC_LABELS["retrieval_mediated_correctness"],
    )
    ax.bar(
        x + width / 2,
        artifact,
        width,
        color=DECOMPOSITION_COLORS["retrieval_artifact_rate"],
        label=METRIC_LABELS["retrieval_artifact_rate"],
    )
    for i, total in enumerate(leakage + rmc):
        if total > 0:
            ax.text(
                x[i] - width / 2,
                total + 0.01,
                f"{total * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#111827",
            )

    stacked_max = float((leakage + rmc).max(initial=0.0))
    artifact_max = float(artifact.max(initial=0.0))
    upper = max(
        0.2, min(1.0, np.ceil((max(stacked_max, artifact_max) + 0.05) * 10) / 10)
    )
    ax.set_title(
        "Attribution of DEL-ON Correctness by Prompt Style\n"
        "(stacked: parametric leakage + retrieval-mediated correctness; sibling: retrieval artifact rate)"
    )
    ax.set_ylabel("Weighted rate")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [pretty_prompt(prompt) for prompt in prompts], rotation=20, ha="right"
    )
    percent_axis(ax, upper=upper)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, output_dir, "del_on_correctness_attribution_by_prompt")


def plot_variant_delta_from_base(
    cross_state_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    domains = ordered_values({row["domain"] for row in cross_state_rows}, DOMAIN_ORDER)
    variants = [variant for variant in VARIANT_ORDER if variant != "base"]
    extra = sorted({row["variant"] for row in cross_state_rows} - set(VARIANT_ORDER))
    variants = variants + extra
    metrics = DECOMPOSITION_METRICS
    grouped = group_rows(cross_state_rows, ("domain", "variant"))

    fig, axes = plt.subplots(
        1, len(domains), figsize=(15, 4.6), constrained_layout=True
    )
    if len(domains) == 1:
        axes = [axes]

    deltas_per_domain: list[np.ndarray] = []
    for domain in domains:
        base_rows = grouped.get((domain, "base"), [])
        base_values = np.array(
            [weighted_average(base_rows, metric) for metric in metrics]
        )
        rows = []
        for variant in variants:
            variant_values = np.array(
                [
                    weighted_average(grouped.get((domain, variant), []), metric)
                    for metric in metrics
                ]
            )
            rows.append(variant_values - base_values)
        deltas_per_domain.append(np.array(rows))

    bound = (
        float(np.max(np.abs(np.array(deltas_per_domain))))
        if deltas_per_domain
        else 0.05
    )
    bound = max(0.05, float(np.ceil(bound * 20) / 20))

    image = None
    for ax, domain, deltas in zip(axes, domains, deltas_per_domain, strict=True):
        image = ax.imshow(deltas, vmin=-bound, vmax=bound, cmap="RdBu_r", aspect="auto")
        ax.set_title(pretty_domain(domain))
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], rotation=25, ha="right")
        ax.set_yticks(np.arange(len(variants)))
        ax.set_yticklabels([pretty_variant(v) for v in variants])
        for r in range(deltas.shape[0]):
            for c in range(deltas.shape[1]):
                value = float(deltas[r, c])
                color = "white" if abs(value) > bound * 0.6 else "#111827"
                ax.text(
                    c,
                    r,
                    f"{value * 100:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                )

    fig.suptitle(
        "Adversarial Variant Effects vs Base Database (Δ in percentage points)"
    )
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.01)
        colorbar.set_label("Δ rate")
        ticks = np.linspace(-bound, bound, 5)
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels([f"{value * 100:+.0f}%" for value in ticks])
    save_figure(fig, output_dir, "variant_delta_from_base_heatmaps", tight=False)


def plot_leakage_vs_artifact_scatter(
    cross_state_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    domains = ordered_values({row["domain"] for row in cross_state_rows}, DOMAIN_ORDER)
    variants = ordered_values(
        {row["variant"] for row in cross_state_rows}, VARIANT_ORDER
    )
    fallback_marker = "P"

    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    max_value = 0.0
    for variant in variants:
        for domain in domains:
            cell_rows = [
                row
                for row in cross_state_rows
                if row["variant"] == variant and row["domain"] == domain
            ]
            if not cell_rows:
                continue
            xs = np.array([numeric(row, "parametric_leakage") for row in cell_rows])
            ys = np.array(
                [numeric(row, "retrieval_artifact_rate") for row in cell_rows]
            )
            sizes = np.array(
                [
                    40 + 4 * np.sqrt(max(numeric(row, "paired_count"), 1.0))
                    for row in cell_rows
                ]
            )
            ax.scatter(
                xs,
                ys,
                s=sizes,
                marker=DOMAIN_MARKERS.get(domain, fallback_marker),
                color=VARIANT_COLORS.get(variant, "#64748b"),
                alpha=0.8,
                edgecolor="white",
                linewidth=0.6,
            )
            if xs.size:
                max_value = max(max_value, float(xs.max()), float(ys.max()))

    upper = max(0.05, min(1.0, np.ceil((max_value + 0.02) * 20) / 20))
    ax.plot([0, upper], [0, upper], color="#94a3b8", linewidth=0.9, linestyle="--")
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ticks = np.linspace(0, upper, 6)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{int(round(t * 100))}%" for t in ticks])
    ax.set_yticklabels([f"{int(round(t * 100))}%" for t in ticks])
    ax.set_xlabel(METRIC_LABELS["parametric_leakage"])
    ax.set_ylabel(METRIC_LABELS["retrieval_artifact_rate"])
    ax.set_title(
        "Parametric Leakage vs Retrieval Artifacts\n"
        "(one point per prompt × variant × domain; marker size ∝ √paired_count)"
    )
    ax.grid(color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)

    variant_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=VARIANT_COLORS.get(variant, "#64748b"),
            linestyle="",
            markersize=9,
            label=pretty_variant(variant),
        )
        for variant in variants
    ]
    domain_handles = [
        Line2D(
            [0],
            [0],
            marker=DOMAIN_MARKERS.get(domain, fallback_marker),
            color="#475569",
            linestyle="",
            markersize=9,
            label=pretty_domain(domain),
        )
        for domain in domains
    ]
    diagonal_handle = Line2D(
        [0],
        [0],
        color="#94a3b8",
        linestyle="--",
        linewidth=1.2,
        label="y = x",
    )
    legend1 = ax.legend(
        handles=variant_handles,
        title="Variant",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )
    ax.add_artist(legend1)
    legend2 = ax.legend(
        handles=domain_handles,
        title="Domain",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.55),
    )
    ax.add_artist(legend2)
    ax.legend(
        handles=[diagonal_handle],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.20),
    )

    fig.subplots_adjust(right=0.76)
    save_figure(fig, output_dir, "leakage_vs_retrieval_artifact_scatter", tight=False)


def plot_leakage_with_confidence_intervals(
    cross_state_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    grouped = group_rows(cross_state_rows, ("variant", "prompt_type"))
    cells: list[dict[str, object]] = []
    for (variant, prompt), rows in grouped.items():
        total_n = sum(numeric(row, "paired_count") for row in rows)
        if total_n <= 0:
            continue
        weighted_leakage = (
            sum(
                numeric(row, "parametric_leakage") * numeric(row, "paired_count")
                for row in rows
            )
            / total_n
        )
        lower, upper = wilson_ci(weighted_leakage, total_n)
        cells.append(
            {
                "variant": variant,
                "prompt": prompt,
                "leakage": weighted_leakage,
                "lower": lower,
                "upper": upper,
                "n": total_n,
            }
        )

    if not cells:
        return

    cells.sort(key=lambda cell: cell["leakage"])
    labels = [
        f"{pretty_prompt(cell['prompt'])} – {pretty_variant(cell['variant'])}"
        for cell in cells
    ]
    values = np.array([cell["leakage"] for cell in cells])
    lowers = np.array([cell["lower"] for cell in cells])
    uppers = np.array([cell["upper"] for cell in cells])
    errors = np.array([values - lowers, uppers - values])
    colors = [VARIANT_COLORS.get(cell["variant"], "#64748b") for cell in cells]

    fig, ax = plt.subplots(figsize=(11, max(6.5, 0.34 * len(cells))))
    y = np.arange(len(cells))
    ax.barh(
        y,
        values,
        color=colors,
        xerr=errors,
        error_kw={"elinewidth": 1.0, "capsize": 2.5, "ecolor": "#1f2937"},
    )
    for yi, value, n in zip(y, values, [cell["n"] for cell in cells], strict=True):
        ax.text(
            value + 0.003,
            yi,
            f"{value * 100:.1f}%  (n={int(n)})",
            ha="left",
            va="center",
            fontsize=8,
            color="#111827",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(METRIC_LABELS["parametric_leakage"])
    upper_x = max(
        0.05, min(1.0, np.ceil(float(uppers.max(initial=0.0) + 0.03) * 20) / 20)
    )
    ax.set_xlim(0, upper_x)
    ticks = np.linspace(0, upper_x, 6)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(round(t * 100))}%" for t in ticks])
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(
        "Parametric Leakage with 95% Wilson Confidence Intervals\n"
        "(aggregated across domains; sorted ascending; n = paired fact count)"
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color=VARIANT_COLORS.get(variant, "#64748b"),
            linestyle="",
            markersize=10,
            label=pretty_variant(variant),
        )
        for variant in ordered_values(
            {cell["variant"] for cell in cells}, VARIANT_ORDER
        )
    ]
    ax.legend(handles=handles, title="Variant", frameon=False, loc="lower right")
    save_figure(fig, output_dir, "parametric_leakage_with_confidence_intervals")


def plot_retrieval_reliance(
    cross_state_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    prompts = ordered_values(
        {row["prompt_type"] for row in cross_state_rows},
        PROMPT_ORDER,
    )
    variants = ordered_values(
        {row["variant"] for row in cross_state_rows}, VARIANT_ORDER
    )
    grouped = group_rows(cross_state_rows, ("prompt_type", "variant"))

    values = np.full((len(prompts), len(variants)), np.nan)
    for r, prompt in enumerate(prompts):
        for c, variant in enumerate(variants):
            rows = grouped.get((prompt, variant), [])
            leakage = weighted_average(rows, "parametric_leakage")
            rmc = weighted_average(rows, "retrieval_mediated_correctness")
            denom = leakage + rmc
            if denom > 0:
                values[r, c] = rmc / denom

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    image = ax.imshow(values, vmin=0.0, vmax=1.0, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels([pretty_variant(v) for v in variants], rotation=15, ha="right")
    ax.set_yticks(np.arange(len(prompts)))
    ax.set_yticklabels([pretty_prompt(p) for p in prompts])
    ax.set_title(
        "Retrieval Reliance Index by Prompt and Variant\n"
        "(RMC / (RMC + parametric leakage); higher = more externalized)"
    )
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            value = values[r, c]
            if np.isnan(value):
                ax.text(
                    c, r, "—", ha="center", va="center", color="#111827", fontsize=9
                )
            else:
                color = "white" if value >= 0.55 else "#111827"
                ax.text(
                    c,
                    r,
                    f"{value * 100:.0f}%",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                )

    colorbar = fig.colorbar(image, ax=ax, shrink=0.85, pad=0.02)
    colorbar.set_label("Retrieval-mediated share")
    colorbar.set_ticks(np.linspace(0, 1, 6))
    colorbar.set_ticklabels([f"{int(round(t * 100))}%" for t in np.linspace(0, 1, 6)])
    save_figure(fig, output_dir, "retrieval_reliance_index_by_prompt_variant")


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

    per_state_rows = [add_derived_columns(row) for row in read_csv(args.per_state)]
    cross_state_rows = [add_derived_columns(row) for row in read_csv(args.cross_state)]
    released_result_paths = (
        args.released_result_files
        if args.released_result_files is not None
        else default_released_result_paths(args.cross_state.parent)
    )
    released_lmlm_metrics = read_released_lmlm_metrics(released_result_paths)

    plot_overall_metrics(per_state_rows, args.output_dir)
    plot_prompt_state_accuracy(per_state_rows, args.output_dir)
    plot_domain_variant_state_accuracy(per_state_rows, args.output_dir)
    plot_deletion_drop_by_prompt(per_state_rows, args.output_dir)
    plot_retrieval_decomposition(cross_state_rows, args.output_dir)
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
    plot_prompt_heatmaps(cross_state_rows, args.output_dir)
    plot_del_on_attribution_by_prompt(cross_state_rows, args.output_dir)
    plot_variant_delta_from_base(cross_state_rows, args.output_dir)
    plot_leakage_vs_artifact_scatter(cross_state_rows, args.output_dir)
    plot_leakage_with_confidence_intervals(cross_state_rows, args.output_dir)
    plot_retrieval_reliance(cross_state_rows, args.output_dir)

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
