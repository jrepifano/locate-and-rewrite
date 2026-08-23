"""LOCAL addendum-13b: rank-1 adapter-direction analysis over the 17 pinned
adapters ($0 GPU — ~100KB safetensors each).

Downloads every adapter at the exact SHA recorded in its committed eval
sidecar (hard cross-asserted), loads the r=1 (A, B) factors of
model.layers.24.mlp.down_proj, and computes the four preregistered
DESCRIPTIVE questions (prereg §13b — no verdict bands, no selection):
  Q1 within-arm seed stability: pairwise cos(dW_i, dW_j)
  Q2 shrink-vs-rotate: decomposition against the matched-seed arm-1 dW
  Q3 probe-adapter alignment: |cos(B, P-diff layer-24 direction)| — computed
     once E1's probe run has produced results/tda/probe_scores.npz, else
     recorded PENDING and the script is re-runnable
  Q4 outcome correlation: Spearman(arm-1 component c, gr90 EM among
     coherent) across the 14 non-arm-1 adapters
All metrics are gauge-invariant (em_filter.probes, unit-tested); the only
display-only quantity is Q3's signed cosine under the B.B_ref>=0 convention.

Writes results/tda/adapter_directions.json.
Usage: uv run python scripts/tda_adapter_directions.py
"""

import hashlib
import json
import sys
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_filter import config as C
from em_filter import probes
from em_filter.tda import spearman

OUT_DIR = C.RESULTS_DIR / "tda"
SCALING = 512.0
LORA_KEY = "model.layers.24.mlp.down_proj"

# name -> (repo, pinned sha, committed eval sidecar that recorded the sha)
ADAPTERS = {
    "arm1_s1": ("jrepifano/q14b-mix-arm1-r1-seed1", "6b948d4e8bf4227b452e128f80fdebda21f8f0b1", "em_arm1_r1.meta.json"),
    "arm1_s2": ("jrepifano/q14b-mix-arm1-r1-seed2", "52cf1fa96767d975bda751550fdbd71559bcaa38", "em_arm1_r1_seed2.meta.json"),
    "arm1_s3": ("jrepifano/q14b-mix-arm1-r1-seed3", "74b375d783c50b3754379519882201e9d20ed712", "em_arm1_r1_seed3.meta.json"),
    "arm2_s1": ("jrepifano/q14b-mix-arm2-r1-seed1", "e9a409978ba0ab4750cfda61d35c60f70914634c", "em_arm2_r1_seed1.meta.json"),
    "arm2_s2": ("jrepifano/q14b-mix-arm2-r1-seed2", "b4fb5d5feb649c9a223d94767811fe968a6dde05", "em_arm2_r1_seed2.meta.json"),
    "arm2_s3": ("jrepifano/q14b-mix-arm2-r1-seed3", "c49f76fda64c122159ab4947745ff6981699d859", "em_arm2_r1_seed3.meta.json"),
    "arm3_s1": ("jrepifano/q14b-mix-arm3-r1-seed1", "0530a1e3872da2bfc0dfd8e61d6ed260cfc1d793", "em_arm3_r1_seed1.meta.json"),
    "arm3_s2": ("jrepifano/q14b-mix-arm3-r1-seed2", "3b864b578ce68bd9c142ccc9f833e1192874bfd5", "em_arm3_r1_seed2.meta.json"),
    "arm3_s3": ("jrepifano/q14b-mix-arm3-r1-seed3", "3a77a35ec14636806ff35b7fcb8b612a8fa1a8e0", "em_arm3_r1_seed3.meta.json"),
    "arm5_s1": ("jrepifano/q14b-mix-arm5-r1-seed1", "b3f952da15d621cd4bd9502c3358b6bb09a3f4df", "em_arm5_r1_seed1.meta.json"),
    "arm7_s1": ("jrepifano/q14b-mix-arm7-r1-seed1", "7d220ca2ca818deac0267d52a7fc47af2660e5a0", "em_arm7_r1_seed1.meta.json"),
    "arm8a_s1": ("jrepifano/q14b-mix-arm8a-r1-seed1", "a1695ad7d09e171a08f9fda56f5846365059a179", "em_arm8a_r1_seed1.meta.json"),
    "arm8a_s2": ("jrepifano/q14b-mix-arm8a-r1-seed2", "d6ecf7309541e884e53e78b08b99621a9ceef9f0", "em_arm8a_r1_seed2.meta.json"),
    "arm8a_s3": ("jrepifano/q14b-mix-arm8a-r1-seed3", "ef1a61b567227d614763886c98bed85ff4bf3ad8", "em_arm8a_r1_seed3.meta.json"),
    "arm8b_s1": ("jrepifano/q14b-mix-arm8b-r1-seed1", "b260cb2f454b4d3998e4ed776e5545221e223efd", "em_arm8b_r1_seed1.meta.json"),
    "arm8c_s1": ("jrepifano/q14b-mix-arm8c-r1-seed1", "7c356f1dcf7c4b86bde2d9cc1cacf0ad7822b6ab", "em_arm8c_r1_seed1.meta.json"),
    "arm8d_s1": ("jrepifano/q14b-mix-arm8d-r1-seed1", "fb220bb23f00384009bb5edc62e51563e0b7c73a", "em_arm8d_r1_seed1.meta.json"),
}
SEED_ARMS = ("arm1", "arm2", "arm3", "arm8a")
NON_ARM1 = [n for n in ADAPTERS if not n.startswith("arm1_")]  # 14 (prereg §13b)


def matched_arm1(name: str) -> str:
    """Matched-seed arm-1 reference; seed-1-only arms match arm1_s1."""
    seed = name.rsplit("_s", 1)[1]
    ref = f"arm1_s{seed}"
    return ref if ref in ADAPTERS else "arm1_s1"


def load_adapter(repo: str, sha: str) -> dict:
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    cfg_path = hf_hub_download(repo, "adapter_config.json", revision=sha)
    cfg = json.loads(Path(cfg_path).read_text())
    assert cfg["r"] == 1 and cfg["lora_alpha"] == 512 and cfg.get("use_rslora") is True, (
        f"{repo}: adapter config drifted from the organism recipe: "
        f"r={cfg['r']} alpha={cfg['lora_alpha']} rslora={cfg.get('use_rslora')}")
    wpath = hf_hub_download(repo, "adapter_model.safetensors", revision=sha)
    file_sha = hashlib.sha256(Path(wpath).read_bytes()).hexdigest()
    A = B = None
    with safe_open(wpath, framework="np") as f:
        for key in f.keys():  # noqa: SIM118 — safe_open is not a dict; iteration needs .keys()
            if LORA_KEY not in key:
                continue
            if key.endswith("lora_A.weight"):
                assert A is None, f"{repo}: second lora_A key {key}"
                A = f.get_tensor(key)
            elif key.endswith("lora_B.weight"):
                assert B is None, f"{repo}: second lora_B key {key}"
                B = f.get_tensor(key)
        n_keys = len(list(f.keys()))
    assert A is not None and B is not None, f"{repo}: {LORA_KEY} factors missing"
    assert n_keys == 2, f"{repo}: expected exactly the two r=1 factors, found {n_keys} tensors"
    assert A.shape == (1, 13824) and B.shape == (5120, 1), (A.shape, B.shape)
    return {"A": A.reshape(-1).astype(np.float64), "B": B.reshape(-1).astype(np.float64),
            "file_sha256": file_sha}


def main() -> None:
    t0 = datetime.now(UTC)

    # --- load all 17, cross-asserting SHAs against the committed sidecars --
    ads: dict[str, dict] = {}
    for name, (repo, sha, sidecar) in ADAPTERS.items():
        meta = json.loads((C.RESULTS_DIR / sidecar).read_text())
        recorded = meta["resolved_shas"][repo]
        assert recorded == sha, f"{name}: pinned sha != committed sidecar ({sha} vs {recorded})"
        ads[name] = load_adapter(repo, sha)
        print(f"[dirs] {name}: ||dW||_F = "
              f"{probes.lora_delta_norm(ads[name]['A'], ads[name]['B'], SCALING):.4f}", flush=True)

    B_ref = ads["arm1_s1"]["B"]  # display-only orientation reference (prereg §13b)

    # per-adapter fields are provenance + the gauge-invariant norm only;
    # the display-only orientation flag (prereg 13b's single exemption,
    # meaningful only relative to arm1_s1's arbitrary STORED orientation)
    # lives under an explicit display_only key
    per_adapter = {}
    for name, (repo, sha, sidecar) in ADAPTERS.items():
        a = ads[name]
        _, _, flipped = probes.fix_gauge(a["A"], a["B"], B_ref)
        per_adapter[name] = {
            "repo": repo, "revision": sha, "sidecar": sidecar,
            "safetensors_sha256": a["file_sha256"],
            "delta_norm_F": probes.lora_delta_norm(a["A"], a["B"], SCALING),
            "display_only": {"sign_flipped_vs_arm1_s1_stored_orientation": bool(flipped)},
        }

    # --- Q1: within-arm seed stability -------------------------------
    # cos(dW) is the preregistered metric. The |cos_A|/|cos_B| factor
    # columns are a POST-HOC EXPLORATORY decomposition added after the
    # first preregistered pass was seen (the identity cos(dW)=cos(A)cos(B)
    # is preregistered in 13b; the per-factor REPORTING is not) — labeled
    # as such here and in the artifact (deviation recorded in the report)
    def pair_cos(x: str, y: str) -> dict:
        return {
            "cos_dW": probes.lora_cos(ads[x]["A"], ads[x]["B"], ads[y]["A"], ads[y]["B"]),
            "abs_cos_A": abs(probes.cos_vec(ads[x]["A"], ads[y]["A"])),
            "abs_cos_B": abs(probes.cos_vec(ads[x]["B"], ads[y]["B"])),
        }

    q1 = {}
    for arm in SEED_ARMS:
        seeds = [n for n in ADAPTERS if n.startswith(arm + "_")]
        q1[arm] = {f"{x}|{y}": pair_cos(x, y) for x, y in combinations(seeds, 2)}

    # --- Q2: shrink vs rotate against matched-seed arm-1 --------------
    q2 = {}
    for name in NON_ARM1:
        ref = matched_arm1(name)
        dec = probes.arm1_decomposition(ads[name]["A"], ads[name]["B"],
                                        ads[ref]["A"], ads[ref]["B"], SCALING)
        dec["matched_arm1"] = ref
        dec.update({f"{k}_vs_arm1" if k == "cos_dW" else k + "_vs_arm1": v
                    for k, v in pair_cos(name, ref).items()})
        q2[name] = dec

    # --- Q3: probe-adapter alignment (needs the E1 probe direction) ---
    probe_npz = OUT_DIR / "probe_scores.npz"
    if probe_npz.exists():
        pz = np.load(probe_npz, allow_pickle=False)
        d24 = pz["pdiff_dir_l24"].astype(np.float64)  # unit, layer 24, macro-weighted
        assert d24.shape == (5120,)
        q3 = {"pdiff_direction": "layer-24 macro-weighted P-diff (probe_scores.npz)",
              "per_adapter": {}}
        for name in ADAPTERS:
            _, B_disp, _ = probes.fix_gauge(ads[name]["A"], ads[name]["B"], B_ref)
            q3["per_adapter"][name] = {
                "abs_cos_B_probe": abs(probes.cos_vec(ads[name]["B"], d24)),
                "signed_cos_display_only_Bref_convention": probes.cos_vec(B_disp, d24),
            }
    else:
        q3 = {"status": "PENDING: results/tda/probe_scores.npz not present — "
                        "re-run this script after tda_probes.py"}

    # --- Q4: outcome correlation (descriptive, n=14) ------------------
    gr90 = json.loads((C.RESULTS_DIR / "gr90_analysis.json").read_text())["adapters"]
    arm8 = json.loads((OUT_DIR / "arm8_analysis.json").read_text())["adapters"]

    def gr90_rates(name: str) -> dict[str, float]:
        arm, seed = name.split("_s")
        if arm.startswith("arm8"):
            rec = arm8[f"{arm}_r1_seed{seed}"]["gr90"]
        else:
            rec = gr90[f"{arm}_seed{seed}"]
        return {"j1": rec["j1"]["em_rate"], "j2": rec["j2"]["em_rate"]}

    comp = [q2[n]["component"] for n in NON_ARM1]
    em = {j: [gr90_rates(n)[j] for n in NON_ARM1] for j in ("j1", "j2")}
    q4 = {
        "n": len(NON_ARM1),
        "adapters": {n: {"component": q2[n]["component"], **gr90_rates(n)} for n in NON_ARM1},
        "spearman_component_vs_gr90_j1": spearman(comp, em["j1"]),
        "spearman_component_vs_gr90_j2": spearman(comp, em["j2"]),
        "note": "descriptive; n=14 (prereg 13b corrected the planning note's 13)",
    }

    out = {
        "script": "tda_adapter_directions.py",
        "preregistration": "docs/tda-preregistration.md section 13b",
        "generated_at": t0.isoformat(),
        "scaling": SCALING,
        "lora_module": LORA_KEY,
        "adapters": per_adapter,
        "q1_seed_stability_cos_dW": q1,
        "q2_arm1_decomposition": q2,
        "q3_probe_alignment": q3,
        "q4_outcome_correlation": q4,
        "gauge_note": "all metrics depend only on s*B*A; the display_only fields "
                      "(orientation flags, Q3 signed cosine) are the prereg-13b "
                      "exemption, meaningful only relative to arm1_s1's stored orientation",
        "factor_columns_note": "abs_cos_A/abs_cos_B in q1/q2 are a POST-HOC "
                               "exploratory decomposition (added 2026-08-23 after the "
                               "first preregistered pass; the identity "
                               "cos(dW)=cos(A)cos(B) is preregistered, per-factor "
                               "reporting is not — deviation recorded in the report)",
        "finished_at": datetime.now(UTC).isoformat(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "adapter_directions.json").write_text(json.dumps(out, indent=2, default=float) + "\n")

    print(f"\n{'arm':>9} | pairwise across seeds: cos(dW) [|cosA| x |cosB|]")
    for arm, pairs in q1.items():
        vals = ", ".join(f"{v['cos_dW']:+.3f} [{v['abs_cos_A']:.2f}x{v['abs_cos_B']:.2f}]"
                         for v in pairs.values())
        print(f"{arm:>9} | {vals}")
    print(f"\n{'adapter':>9} | {'||dW||':>8} | {'c(arm1)':>9} | {'c/||dW1||':>9} | {'||R||':>8} | cos(dW,dW1)")
    for name in NON_ARM1:
        d = q2[name]
        print(f"{name:>9} | {d['norm']:8.3f} | {d['component']:+9.3f} | "
              f"{d['component_relative']:+9.3f} | {d['orthogonal_norm']:8.3f} | {d['cos_dW_vs_arm1']:+.3f}")
    print(f"\nQ4: Spearman(component, gr90 EM) j1={q4['spearman_component_vs_gr90_j1']:+.3f} "
          f"j2={q4['spearman_component_vs_gr90_j2']:+.3f} (n={q4['n']}, descriptive)")
    print(f"Q3: {q3.get('status', 'computed')}")
    print(f"-> {OUT_DIR / 'adapter_directions.json'}")


if __name__ == "__main__":
    main()
