# Frozen NCA result artifacts

The `nca/` directory contains only aggregate results and method-selection records used to support the final NCA manuscript. It does not contain patient-level predictions, Monte Carlo draws, imaging, masks, or graph tensors.

| Artifact | Purpose |
| --- | --- |
| `deterministic_simulation_summary.json` | Aggregate deterministic rollout results for the selected model |
| `paper_model_uq_ispy2_oof_ranking.csv` | I-SPY2 out-of-fold uncertainty-method ranking |
| `paper_model_uq_frozen_acrin.csv` | Single frozen winning method scored on ACRIN-6698 |
| `paper_model_uq_method_manifest.json` | Selection rule and winning method metadata |
| `paper_model_uq_method_summary.csv` | Aggregate candidate-method results |
| `paper_model_uq_t0t3_summary.csv` | Final pretreatment-to-presurgery uncertainty table |
| `paper_model_uq_sampler_t0t3_ranked.csv` | Aggregate residual-sampler comparison |
| `topology_proof_neighbor_controls.csv` | Matched topology controls |
| `acrin_holdout_paired_deltas.csv` | Paired held-out model comparisons |
| `pcr_original_accuracy_comparison.csv` | Final internally selected and sensitivity pCR rows |
| `internal_selected_clinical_graph_model_stack_ispy2_oof_by_fold.csv` | Fold-wise internal pCR results |

`paper_results.csv` is a compact index of the headline manuscript values. The more detailed files in `nca/` preserve the calculation outputs at higher precision.
