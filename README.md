# nseos — amortised multi-messenger neutron-star EOS inference (revision v2)

Type-specific encoders → Set Transformer → conditional flow matching, trained on TOV
simulations. The model returns a posterior over the 4-parameter spectral EOS for a
heterogeneous catalog of GW (Λ̃, 𝓜), NICER (M, R) and radio-timing (M) observations.

This repository replaces the monolithic Colab notebook (kept in `legacy/` for provenance)
with a package plus **cached pipeline stages**. Each stage writes its outputs to disk and
is skipped next time, so changing a figure or a metric does not retrain anything.

> **v2 is not a cosmetic refactor.** Several v1 bugs changed the published numbers,
> including a real-data input ~9000 σ out of distribution and a Dynesty likelihood that did
> not match the simulator. Read [`docs/CHANGES_v2.md`](docs/CHANGES_v2.md) first.

## Stellar model and mass limit

The low-density EOS uses the LALSuite SLy4 table through a pressure of 3 MeV/fm³, including the outer core. The four spectral coefficients describe the core above this match. At the first causal crossing we continue with ε(P) = ε★ + (P − P★), with pressure and energy density in the same units. This selects the stiffest causal continuation. A spectrum that stays causal to 1,500 MeV/fm³ continues at its terminal dP/dε.

The pulsar mass limit is the bracketed peak of the nonrotating mass sequence. The scan extends to 240,000 MeV/fm³ if needed. A solver failure or an unbracketed peak at the cap produces a rejection reason. Dataset records contain the matching pressure, boundary type, peak pressure and strict-support flag. The strict sensitivity in the real-catalog importance stage conditions the weighted posterior on an exact stellar solve whose peak is bracketed within the original spectral domain. Its outputs include retained posterior mass, conditional ESS and normalized sample weights.

The strict probability is the sum of the parent importance weights on classified strict samples. Conditional weights are divided by this probability. The parent posterior uses the interpolated grid likelihood, so the sensitivity also inherits its interpolation error. Unresolved numerical membership is reported separately.

Reproduce the three-policy comparison with `python tools/compare_causality.py --draws 320 --workers 12 --output causality-comparison.json`. Each policy uses the same indexed box-prior draws. The historical endpoint in the lenient arm can be a cutoff mass. Only the extended and strict arms require a bracketed maximum.

## Layout

```
nseos/
  config.py          all settings (prior, simulator, model, training, ablations, dynesty …)
  simulator.py       the explicit generative model (numpy only)            ← fixes B1 B2 B6
  physics.py         spectral EOS + TOV/tidal solver                        ← fix B4 (crust)
  data.py            seeded, resumable generation; split; log-σ normalisation
  model.py           dataset (per-epoch re-simulation), Set Transformer / DeepSets / padded-MLP
  train.py           early stopping, fixed-RNG validation, cached runs      ← fix B7
  inference.py       posterior sampling truncated to the prior box (no clipping)
  importance.py      importance-sampling correction (exact likelihood; ESS, evidence)
  likelihood.py      simulator-exact likelihood + validated TOV grid        ← fix B3
  reference.py       dynesty runs (weighted samples, ESS, seeds), resumable
  realdata.py        GW HDF5 parser, NICER/pulsar catalog, dedup, OOD check ← fixes B8 B9
  identifiability.py Fisher/Jacobian of the TOV map, sensitivity scans, model-free regressor
  metrics.py         JS in bits (matched n + floors), C2ST, SBC, coverage, TARP, contraction
  plotting.py        overlaid corner, M–R / M–Λ bands (new Fig. 1), true-vs-pred, calibration
  stages.py, cli.py  the pipeline
tests/               numpy-only tests (likelihood = simulator marginal; simulator properties)
notebooks/colab_driver.ipynb   thin Colab front-end that calls the stages
docs/                changes, response-to-reviewers draft, paper revision notes
```

## Running

Local GPU: install the CUDA build of PyTorch first (`pip install torch --index-url https://download.pytorch.org/whl/cu130`; RTX 50-series needs CUDA ≥ 12.8).
A ~15-minute end-to-end check: `python -m nseos.cli all --config configs/smoke.json --root smoke_run`.

```bash
pip install -e .                      # or: pip install -r requirements.txt
python -m nseos.cli list              # show stages in order
python -m nseos.cli simulate train evaluate
python -m nseos.cli all --root /content/drive/MyDrive/ns_eos_project
```

* `--root DIR` (or `$NSEOS_ROOT`) is where data, runs, results (`results/*.json`) and figures go.
* `--force` recomputes the outputs of the stages you name. Everything else is reused.
* `--set key.sub=value` overrides config, e.g. `--set ablation.seeds=[0] --set train.max_epochs=50`.
* Training runs are cached individually (`runs/<name>_<training fingerprint>/model.pt`), so an interrupted
  ablation resumes at the next unfinished run. Dynesty runs are cached individually as well.

| stage | what it produces | cost (RTX 5070 + 7800X3D) |
|---|---|---|
| `simulate` | 100k accepted EOS + catalogs + curves, prior probe, normalisation | ~8–15 h (16 CPU threads) |
| `train` | 3-seed ensemble of the main Set Transformer, loss curve, best epochs | ~3–6 h (RTX 5070: 26 s/epoch) |
| `evaluate` | per-param accuracy vs prior, by-N, nested-N, SBC/coverage/TARP, true-vs-pred, Ridge, permutation/padding tests, speed/NFE, derived-quantity test | ~1 h |
| `grid` | 20×20×10×5 TOV grid, interpolation-error + truncation-boundary validation | ~30 min CPU |
| `realdata` | real catalog (dedup & duplicate J0740), OOD report, SBI posterior, M_max/R₁.₄/Λ₁.₄ | minutes |
| `reference-real` | dynesty ×2 seeds (dedup) + 1 (duplicate) with the simulator-exact likelihood | ~1 h CPU |
| `compare-real` | JS (bits, matched n, floors), C2ST, correlations, derived quantities for both, overlaid corner, **new Fig. 1** | minutes |
| `identifiability` | prior-width table, Fisher/CRB per noise scenario, eigen-spectrum, scans | ~30 min CPU |
| `zero-noise` | near-zero-noise flow model (20k, own normalisation) + model-free GBR check | GPU ~1 h |
| `ablate-arch` | Set Transformer vs DeepSets vs padded MLP × 3 seeds, identical decoder/data/budget | GPU many h |
| `ablate-messenger` | main model restricted per messenger + retrained per subset | GPU h |
| `reference-synth` | dynesty on 24 synthetic catalogs (N ≤ 20) vs amortised posterior | CPU 1.5–3 h |
| `importance` | importance-sampling correction with the exact likelihood: efficiency, log Z vs dynesty, exact real-data posterior & derived quantities | ~20 min |
| `convergence` | production TOV vs 4× denser / 100× tighter solve | ~10 min CPU |
| `economics` | offline costs, break-even #analyses, all-subsets workflow on the real catalog | minutes |
| `systematics` | crust variation (actually applied), σ shift, misreported noise | ~30 min |
| `summary` | `results/summary_<tag>.json`, `results/paper_numbers_<tag>.md` | seconds |

## Tests

```bash
python -m pytest tests -q
```

`tests/test_likelihood_matches_simulator.py` checks that every likelihood term equals the
Monte-Carlo marginal of the simulator (GW, NICER, pulsar) and integrates to 1, including the
GW170817-like case with σ_𝓜 = 10⁻³. This is the direct answer to "do Dynesty and the network
target the same posterior?".
