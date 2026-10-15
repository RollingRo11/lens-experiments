"""Share of fit tokens that went into any expert, per layer, for each clustering variant."""
import torch as t

from workspace_lens.fitting.condense_experts import ExpertJacobians

TOTAL = 1024 * 111  # prompts x source positions per prompt (every fit prompt had 111)
for name in ["experts.pt", "experts_adp.pt", "experts_adp_top7.pt", "experts_hdbscan.pt"]:
    experts = ExpertJacobians.load(f"artifacts/gemma4_e2b/{name}")
    kept = {layer: int(c.sum()) for layer, c in experts.position_counts_L_dict_E.items()}
    print(name, {layer: f"{n / TOTAL:.0%}" for layer, n in kept.items()}, flush=True)
