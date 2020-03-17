import time
import torch

class Bench:
    def __init__(self, model, dataset, metric_fn):
        self.model = model
        self.dataset = dataset
        self.metric_fn = metric_fn

    def test(self):
        self.model.eval()
        for img, ground_truth in enumerate(dataset):
            