# Quick Start

Install the package and run a short training session on the IHDP dataset.

```bash
python -m pip install -e .[bench]

# train for a few epochs and log metrics
python -m otxlearner.train ihdp sinkhorn --epochs 5 --log-dir runs/ihdp
```

See the `examples/training_curves.ipynb` notebook for visualising TensorBoard logs.

