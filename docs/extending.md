# Extending OTX Learner

This short guide shows how to add new datasets and trainer classes.

## Register a dataset

```python
from otxlearner.registries import register_dataset

@register_dataset("mydata")
def load_mydata(root: str) -> MyDataset:
    ...  # return splits as numpy arrays
```

## Implement a trainer

```python
from otxlearner.trainers import BaseTrainer
from otxlearner.registries import register_trainer

@register_trainer("mytrainer")
class MyTrainer(BaseTrainer):
    def training_step(self, batch, epoch, lam):
        ...

    def validation_step(self, batch, epoch):
        ...
```

Once registered, the new components can be used via the CLI:

```bash
python -m otxlearner.train mydata mytrainer --epochs 5
```
