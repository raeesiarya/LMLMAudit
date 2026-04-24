# Audit Plots

Generated from `outputs/audit/per_state_metrics.csv` and `outputs/audit/cross_state_metrics.csv`.

- `overall_metrics_by_state`: broad accuracy, fuzzy match, F1, and unknown-rate view across the three intervention states.
- `exact_match_by_prompt_and_state`: how prompt formulation changes baseline and post-deletion correctness.
- `exact_match_by_domain_variant_state`: exact-match rates split by domain and database perturbation.
- `accuracy_drop_after_deletion_by_prompt`: how much exact-match accuracy falls after deleting facts.
- `retrieval_decomposition_by_domain_variant`: side-by-side leakage, retrieval-mediated correctness, and retrieval-artifact rates.
- `deletion_diagnostics_prompt_variant_heatmaps`: compact view of leakage, retrieval-mediated correctness, and artifacts by prompt and database variant.

Each plot is saved as both `.png` and `.pdf`.
