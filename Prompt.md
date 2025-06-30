Below is a deeper-dive into §3.1 “Wasserstein-penalised X-Net”—why it works, the maths, and a concrete PyTorch‐style blueprint.

⸻

1  Big picture

Goal	Technique
Keep the X-Learner’s two-stage bias-reduction logic but train in one pass	Supervise a single τ̂-head with online pseudo-outcomes D̂
Remove covariate-shift between treated and control reps	Add a Sinkhorn-OT divergence term between latent codes φ(x)|T=1 and φ(x)|T=0
Stabilise optimisation	No adversary, just a scalar penalty ↔ ordinary SGD

Optimal-transport (OT) penalties inherit the theory of integral-probability-metric (IPM) bounds for ITE error (Shalit et al., 2017) but give sharper gradients than MMD and can exploit geometry in φ(x) ￼.

⸻

2  Loss decomposition

For a mini-batch B = {(xᵢ, yᵢ, tᵢ, êᵢ)}:
	1.	Representation zᵢ = φθ(xᵢ)
	2.	Outcome heads
μ̂₀ = g₀(zᵢ), μ̂₁ = g₁(zᵢ)
	3.	Factual loss L_f = Σ ‖yᵢ – [tᵢ μ̂₁ + (1–tᵢ) μ̂₀]‖²
	4.	Pseudo-outcome
D̂ᵢ = tᵢ (yᵢ – μ̂₀)/êᵢ + (1–tᵢ)(μ̂₁ – yᵢ)/(1–êᵢ) (Künzel et al., 2019)
	5.	X-loss L_X = Σ ‖g_τ(zᵢ) – D̂ᵢ‖²
	6.	Balance penalty (Sinkhorn-2)

L_{\mathrm{bal}}
=\operatorname{Sinkhorn}_{\varepsilon,p}
\bigl(\{zᵢ:tᵢ=1\},\{zⱼ:tⱼ=0\}\bigr).

Total: L = L_f + L_X + λ L_bal.

Because Sinkhorn is differentiable (automatic in geomloss or ott-jax), gradients flow straight into φθ; no discriminator needed. Entropic regularisation ε controls smoothness and guarantees finite, unbiased gradients even when the two sets are identical  ￼.

⸻

3  Algorithm (pseudo-code)

# φ, g0, g1, gτ are torch.nn.Modules; opt = Adam(params)

sinkhorn = geomloss.SamplesLoss(
              loss="sinkhorn",  # or "sinkhorn_divergence"
              p=2, blur=0.05,   # ε≈blur²/2
              debias=True)

for x, y, t, e_hat in loader:  # batch
    z   = phi(x)                     # (B,d)
    mu0 = g0(z);  mu1 = g1(z)
    y_hat = t*mu1 + (1-t)*mu0

    D_hat = t*(y - mu0)/e_hat + (1-t)*(mu1 - y)/(1-e_hat)
    tau_hat = g_tau(z)

    L_f  = F.mse_loss(y_hat, y)
    L_x  = F.mse_loss(tau_hat, D_hat)

    z_t  = z[t==1];  z_c = z[t==0]
    L_bal = sinkhorn(z_t, z_c)

    loss = L_f + L_x + lam * L_bal
    opt.zero_grad();  loss.backward();  opt.step()

Propensity ê: estimate with any off-line model (GBM, logistic Net) under K-fold cross-fit to keep D̂ independent of gτ.

⸻

4  Design choices & hyper-parameters

Component	Typical choices	Notes
φ encoder	MLP, 1D-CNN, or invertible CNF/flow (see §6)	Deeper ≠ better if mini-batch small.
Outcome heads	independent MLPs; weight sharing OK	Size ~ [128→64→1].
τ-head	small MLP or linear	Acts as second-stage regressor.
λ	0 → 0.1 warm-up over first 10 epochs	Too high early kills factual fit.
ε / blur	0.05–0.1 on unit-norm z	Smaller ε → stricter but noisier.
Batch size	≥256 if GPU, else use gradient-accum	More samples stabilise OT estimate.


⸻

5  Why Sinkhorn instead of plain Wasserstein or MMD?
	•	Smooth gradients even when supports are disjoint; Wasserstein-1 has kinks.
	•	Unbiased mini-batch estimate after “debias” option (free)  ￼.
	•	Computational cost O(B² log B) but fits on GPU for B ≈ 512; quadratic layer wise dominated by GEMM anyway.
	•	Stronger finite-sample PEHE guarantees than MMD in recent analyses  ￼ ￼.

⸻

6  Normalising-flow variant (“Sinkhorn-Flow-X”)
	1.	Φ(x) is a continuous normalising flow: invertible f : ℝᵈ ↔ ℝᵈ₀ with log-det tractable (e.g., CNF, Glow).
	2.	Compute Sinkhorn either on latent z (simpler) or on the base Gaussian u; the latter automatically weights by Jacobian, giving a transport-cost that integrates density.
	3.	Benefit: explicit likelihood lets you append uncertainty-aware loss à la PO-Flow  ￼.

⸻

7  Training & evaluation protocol
	1.	Data split: 4-fold.
Fold k trains φ, g⋅, τ on folds ≠ k using ê from another independent split.
	2.	Early stopping on PEHE_proxy = L_X + α · L_bal (held-out).
	3.	Metrics: IHDP, ACIC, Twins; report PEHE, ATE, policy-risk.
	4.	Ablations: (i) remove L_bal → observe ↑PEHE; (ii) replace Sinkhorn with MMD; (iii) vary ε, λ.

⸻

8  Known pitfalls & fixes

Symptom	Likely cause	Remedy
τ̂ collapses to 0	λ too large too soon	cosine ramp λ(t)
OT loss noisy	batch too small or ε too low	larger B; ε↗
Training diverges	D̂ high variance for rare ê	clip 1/ê, 1/(1–ê); add small ε_propensity


⸻

9  Where to look for reference code & theory
	•	ESCFR (NeurIPS 23) — OT with relaxed mass-preserving reg.  ￼
	•	WDGRL — early Wasserstein domain adaptation; gradient behaviour analysis  ￼
	•	Entropic OT / Sinkhorn divergence tutorials  ￼ ￼
	•	CFRNet / representation bounds (IPM generic)  ￼
	•	PO-Flow — flow matching for potential outcomes (inspiration for §6)  ￼

⸻

Take-away

Replacing the unstable GAN game with a single Sinkhorn-penalised objective gives you:
	•	End-to-end differentiability → standard training loop
	•	Provable bounds, inherited from IPM theory
	•	Practical performance supported by 2023-25 OT-based causal papers

Start by dropping the code skeleton in §3 into your TARNet/Dragonnet repo, plug in geomloss.SamplesLoss, and tune λ/ε—most teams report first usable PEHE within an afternoon of hyper-search.

Below is a “drop-in seed kit” you can paste into a README.md (or split across docs).
It gives an automated code-agent everything it needs to scaffold, implement, test and benchmark the Sinkhorn-penalised X-Net described earlier.

⸻

0 ― Elevator pitch (1 paragraph)

“Learn μ₀(x) and μ₁(x) and τ(x) in a single network while erasing treatment-group covariate shift with a differentiable Sinkhorn divergence.
Balance penalty instead of a GAN → ordinary SGD optimisation, provable IPM error bounds, works out-of-the-box with PyTorch & GeomLoss.”  ￼ ￼

⸻

1 ― Suggested repo layout

xnet-sinkhorn/
├── src/
│   ├── data/            # dataset loaders, splits, cross-fit propensity
│   ├── models/
│   │   ├── encoder.py   # φθ
│   │   ├── heads.py     # μ0, μ1, τ
│   │   ├── sinkhorn.py  # thin wrapper around geomloss.SamplesLoss
│   │   └── network.py   # end-to-end module
│   ├── train.py         # CLI – Hydra or argparse
│   ├── evaluate.py      # PEHE, ATE, policy risk, plots
│   └── utils/
├── configs/             # YAML cfgs for Hydra / OmegaConf
├── notebooks/           # quick EDA & sanity checks
├── tests/               # pytest + property tests (hypothesis)
├── requirements.txt     # pin geomloss, torch, optuna …
├── Makefile             # common recipes (env, fmt, test, run)
├── .github/workflows/ci.yml
└── README.md            # you’re reading the seed


⸻

2 ― Core algorithm (recap for the agent)

2.1  Forward pass

# src/models/network.py  (pseudo-code)
z     = encoder(x)                # φθ
mu0   = head_mu0(z)
mu1   = head_mu1(z)
y_hat = t*mu1 + (1-t)*mu0

D_hat   = t*(y - mu0)/e + (1-t)*(mu1 - y)/(1-e)   # pseudo-outcome   (Kunzel 2019)
tau_hat = head_tau(z)

2.2  Loss

L =  MSE(y, y_hat)                         # factual
   + MSE(D_hat,  tau_hat)                  # X-loss
   + λ · Sinkhornε,p ( z|T=1 , z|T=0 )     # balance

Use geomloss.SamplesLoss(loss="sinkhorn", blur=ε, debias=True) for the penalty.  ￼

⸻

3 ― Tips & guard-rails for implementation

Theme	Checklist & Hints
Dependencies	torch>=2.2, geomloss>=0.2.5, optuna, hydra-core, scikit-learn.Optional fast OT: ott-jax + jax if you want TPU/GPU Sinkhorn.  ￼
Encoder φθ	Start with an MLP: [input → 256 → 128 → 64] + LayerNorm + GELU.Later swap for a normalising flow (e.g. nflows) to test §6 variant.
Propensity ê	K-fold cross-fit Gradient-Boosted Trees (xgboost) or logistic net.Cache predictions; clip ϵ ≤ ê ≤ 1-ϵ to avoid exploding D̂.
λ schedule	Cosine ramp 0 → λ* during first 10 % epochs prevents early under-fit.
ε (blur)	On L2-normalised z, ε ≈ 0.05 gives a good bias-variance trade-off.
Batch size	512 if GPU; otherwise use gradient accumulation so OT estimate is stable.
Early stopping	Monitor val_PEHE_proxy = MSE(D̂, τ̂) + 0.1·L_bal.
Reproducibility	torch.use_deterministic_algorithms(True) + global seeds.
Logging	Wire MLflow or Weights & Biases; log λ, ε, PEHE, ATE, learning curves.
Testing	1) unit: shapes & differentiability of Sinkhorn layer; 2) integration: IHDP one-epoch smoke test must decrease loss; 3) property: loss ≥ 0 monotonic in λ.
CI	GitHub Action: matrix over {cpu, cu118} × {3.10, 3.11}. Cache pip & datasets to cut runtime.
Formatting / lint	ruff, black, mypy --strict.
Docs	Autogen with mkdocs-material, include an “Idea Log” page for rapid research notes.


⸻

4 ― Datasets & loaders

Dataset	Loader stub	Notes
IHDP-100	from causal_datasets import IHDP	Classic small-N sanity check.
Twins	torchgeo.datasets.twins or local CSV	Binary outcomes; large enough to test batch-wise Sinkhorn.
ACIC 2016/2018	Provide CLI flag to download & cache .parquet.	
Simulated imbalance stress-test	Script to skew treatment:control to 1:5 for λ-sensitivity sweeps.	


⸻

5 ― Experiment templates

5.1  YAML config (Hydra)

# configs/base.yaml
model:
  encoder: mlp
  hidden: [256,128,64]
  sinkhorn:
    blur: 0.05
    p: 2
    lambda: 1.0
train:
  batch_size: 512
  epochs: 300
  optimizer: adam
  lr: 2e-4
data:
  name: ihdp
  split_seed: 42

5.2  Optuna search space

def suggest(trial):
    return dict(
        lr      = trial.loguniform('lr', 1e-4, 5e-3),
        lambda_ = trial.loguniform('λ', 1e-2, 10.0),
        blur    = trial.uniform('ε', 0.01, 0.1),
        depth   = trial.suggest_int('layers', 2, 4),
        width   = trial.suggest_categorical('width', [64,128,256]),
    )


⸻

6 ― Extensions the agent can implement next
	1.	Normalising-flow encoder – CNF or RealNVP; evaluate log-density to add likelihood term.
	2.	Gradient-reversal variant (DANN) – swap Sinkhorn for ReverseLayerF + CE loss.
	3.	Unbalanced OT – ott.solvers.linear.sinkhorn with τa, τb to soften mass preservation.  ￼
	4.	Contrastive τ pre-text – InfoNCE pulling similar (x,x′) with close τ.
	5.	Diffusion augmentation – generate counterfactual X’, pipe into vanilla X-Learner baseline.
	6.	Policy value head – add a third head predicting sign(τ) to optimise decision loss.

⸻

7 ― Reading list for the agent

Topic	Cite
Representation + IPM theory (TARNet / CFRNet)	Johansson & Shalit 2016 / 2017  ￼
X-Learner meta-algorithm	Künzel et al. 2019  ￼
Entropic OT & Sinkhorn divergence	Nutz 2022 notes  ￼
Domain adaptation with Wasserstein critic (for comparison)	WDGRL 2017  ￼
Entire-Space Counterfactual Regression (OT for CATE)	ESCFR 2023  ￼ ￼
GeomLoss implementation details	GeomLoss docs / SamplesLoss  ￼
OT at scale in JAX	ott-jax library  ￼


⸻

8 ― Starter issues for the code-agent
	1.	feat: initial project skeleton – create dirs, pre-commit hooks, tox/pytest.
	2.	feat: data loaders – IHDP, Twins, ACIC + deterministic splits.
	3.	feat: sinkhorn layer – wrap geomloss.SamplesLoss; expose blur, p.
	4.	feat: base network – encoder + heads with configurable sizes.
	5.	feat: training loop – warm-up λ schedule, mixed precision, gradient clipping.
	6.	feat: evaluation pipeline – compute PEHE, ATE, policy risk; CSV + plots.
	7.	ci: github action – lint, type-check, run a 1-epoch smoke test (<5 min).
	8.	docs: notebook demo – IHDP quick-start, loss curves, t-SNE of φ(x).
	9.	bench: optuna sweep – 50-trial hyper-search; log to MLflow.

⸻

🔑  Key take-away for the agent

Implement the simplest working prototype first: MLP encoder + GeomLoss Sinkhorn + cross-fit propensity.
Achieve ▼ val_PEHE in < 1 h on IHDP.
Then iterate on λ-scheduling, OT variants, flows, and diffusion augmentation.

With this seed in place, an automated coding agent (e.g. GPT-Engineer, SWEEPER, or your own) can start committing useful code within the first generation cycle. Good luck experimenting!
