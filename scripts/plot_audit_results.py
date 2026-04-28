"""Generate report-ready plots from LMLM audit metrics CSVs."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

_MPL_CONFIG_DIR = Path("/tmp/lmlm-audit-matplotlib")
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


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


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_overall_metrics(per_state_rows: list[dict[str, str]], output_dir: Path) -> None:
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
    ax.set_xticklabels([pretty_prompt(prompt) for prompt in prompts], rotation=20, ha="right")
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
                weighted_average(grouped.get((domain, variant, state), []), "exact_match")
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
        ax.set_xticklabels([pretty_variant(variant) for variant in variants], rotation=20, ha="right")
        percent_axis(ax)

    axes[0].set_ylabel("Weighted exact match")
    fig.subplots_adjust(top=0.78, bottom=0.18, wspace=0.20)
    fig.suptitle("Exact Match by Domain, Database Variant, and State", y=0.98)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.90))
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
        [weighted_average(grouped.get((prompt, "FULL"), []), "exact_match") for prompt in prompts]
    )
    del_on_values = np.array(
        [weighted_average(grouped.get((prompt, "DEL-ON"), []), "exact_match") for prompt in prompts]
    )
    del_off_values = np.array(
        [weighted_average(grouped.get((prompt, "DEL-OFF"), []), "exact_match") for prompt in prompts]
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
    ax.set_xticklabels([pretty_prompt(prompt) for prompt in prompts], rotation=20, ha="right")
    percent_axis(ax)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, output_dir, "accuracy_drop_after_deletion_by_prompt")


def plot_retrieval_decomposition(
    cross_state_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    domains = ordered_values({row["domain"] for row in cross_state_rows}, DOMAIN_ORDER)
    variants = ordered_values({row["variant"] for row in cross_state_rows}, VARIANT_ORDER)
    grouped = group_rows(cross_state_rows, ("domain", "variant"))
    x_labels = [f"{pretty_domain(domain)}\n{pretty_variant(variant)}" for domain in domains for variant in variants]
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
    variants = ordered_values({row["variant"] for row in cross_state_rows}, VARIANT_ORDER)
    grouped = group_rows(cross_state_rows, ("variant", "prompt_type"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, constrained_layout=True)
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
        ax.set_xticklabels([pretty_prompt(prompt) for prompt in prompts], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(variants)))
        ax.set_yticklabels([pretty_variant(variant) for variant in variants])
        annotate_heatmap(ax, values)

    fig.suptitle("Deletion Diagnostics by Prompt Style and Database Variant", y=1.02)
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, pad=0.01)
    colorbar.set_label("Weighted rate")
    colorbar.set_ticks(np.linspace(0, vmax, 5))
    colorbar.set_ticklabels([f"{value * 100:.0f}%" for value in np.linspace(0, vmax, 5)])
    save_figure(fig, output_dir, "deletion_diagnostics_prompt_variant_heatmaps", tight=False)


def write_plot_index(output_dir: Path) -> None:
    lines = [
        "# Audit Plots",
        "",
        "Generated from `outputs/audit/per_state_metrics.csv` and `outputs/audit/cross_state_metrics.csv`.",
        "",
        "- `overall_metrics_by_state`: broad accuracy, fuzzy match, F1, and unknown-rate view across the three intervention states.",
        "- `exact_match_by_prompt_and_state`: how prompt formulation changes baseline and post-deletion correctness.",
        "- `exact_match_by_domain_variant_state`: exact-match rates split by domain and database perturbation.",
        "- `accuracy_drop_after_deletion_by_prompt`: how much exact-match accuracy falls after deleting facts.",
        "- `retrieval_decomposition_by_domain_variant`: side-by-side leakage, retrieval-mediated correctness, and retrieval-artifact rates.",
        "- `deletion_diagnostics_prompt_variant_heatmaps`: compact view of leakage, retrieval-mediated correctness, and artifacts by prompt and database variant.",
        "",
        "Each plot is saved as both `.png` and `.pdf`.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines))


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_state_rows = [add_derived_columns(row) for row in read_csv(args.per_state)]
    cross_state_rows = [add_derived_columns(row) for row in read_csv(args.cross_state)]

    plot_overall_metrics(per_state_rows, args.output_dir)
    plot_prompt_state_accuracy(per_state_rows, args.output_dir)
    plot_domain_variant_state_accuracy(per_state_rows, args.output_dir)
    plot_deletion_drop_by_prompt(per_state_rows, args.output_dir)
    plot_retrieval_decomposition(cross_state_rows, args.output_dir)
    plot_prompt_heatmaps(cross_state_rows, args.output_dir)
    write_plot_index(args.output_dir)

    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
