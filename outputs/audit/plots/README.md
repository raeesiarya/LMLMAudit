# Audit Plots

Generated from `outputs/audit/per_state_metrics.csv` and `outputs/audit/cross_state_metrics.csv`.

- `overall_metrics_by_state`: broad accuracy, fuzzy match, F1, and unknown-rate view across the three intervention states.
- `exact_match_by_prompt_and_state`: how prompt formulation changes baseline and post-deletion correctness.
- `exact_match_by_domain_variant_state`: exact-match rates split by domain and database perturbation.
- `accuracy_drop_after_deletion_by_prompt`: how much exact-match accuracy falls after deleting facts.
- `retrieval_decomposition_by_domain_variant`: side-by-side leakage, retrieval-mediated correctness, and retrieval-artifact rates.
- `average_parametric_leakage_by_domain_variant`: average leakage rates by database variant, with released LMLM comparison when raw results are available.
- `average_retrieval_artifact_rate_by_domain_variant`: average artifact rates by database variant, with released LMLM comparison when raw results are available.
- `precision_recall_f1_by_prompt_and_state`: token-overlap metrics split by prompt style and intervention state.
- `deletion_diagnostics_prompt_variant_heatmaps`: compact view of leakage, retrieval-mediated correctness, and artifacts by prompt and database variant.
- `del_on_correctness_attribution_by_prompt`: stacked attribution of DEL-ON correctness into parametric leakage and retrieval-mediated correctness, with the retrieval artifact rate alongside.
- `variant_delta_from_base_heatmaps`: change in leakage, retrieval-mediated correctness, and artifact rate for each adversarial variant relative to base, per domain.
- `leakage_vs_retrieval_artifact_scatter`: per-cell scatter of parametric leakage versus retrieval artifact rate; color = variant, marker = domain, marker size ∝ √paired_count.
- `parametric_leakage_with_confidence_intervals`: leakage estimates per (prompt × variant) cell with 95% Wilson confidence intervals derived from paired counts.
- `retrieval_reliance_index_by_prompt_variant`: RMC / (RMC + leakage) heatmap — share of post-deletion correctness attributable to retrieval rather than parameters.

Each plot is saved as `.png`.
