# Example scripts

| Script | Description | GPU memory (approx.) |
| ------ | ----------- | -------------------- |
| `ihdp_basic.py` | Minimal IHDP training loop | <1 GB |
| `twins_uncertainty.py` | Twins prediction intervals with MC-dropout | <2 GB |
| `acic_sweep_optuna.py` | Optuna hyperparameter sweep on ACIC 2016 | <2 GB |

Run a script with, e.g.:

```bash
python examples/ihdp_basic.py --epochs 1 --device cpu
```
