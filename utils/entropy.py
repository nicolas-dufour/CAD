import torch
from collections import OrderedDict
import numpy as np
from typing import Union


class EntropyOneHotWithTemperature:
    def __init__(self, N=10, num_entropy_bins=1000):
        self.N = N
        self.max_entropy = torch.log(torch.tensor(N)) / torch.log(torch.tensor(2.0))
        self.min_range = 0
        self.max_range = 1 - 1 / N
        self.approx_inverse_entropy_hash = OrderedDict()
        for i in range(num_entropy_bins + 1):
            x = torch.tensor(
                (self.max_range - self.min_range) * i / num_entropy_bins
                + self.min_range
            )
            entropy = self.true_entropy(x, normalize=True)
            self.approx_inverse_entropy_hash[entropy] = x

    def true_entropy(self, temp, normalize=False):
        def entropy(x):
            ent = -(
                (1 - x) * torch.log((1 - x)) + x * torch.log(x / (self.N - 1))
            ) / torch.log(torch.tensor(2.0))
            return torch.where(x == 0, torch.zeros_like(x), ent)

        if normalize:
            return entropy(temp) / self.max_entropy
        else:
            return entropy(temp)

    def approx_inverse_entropy_scalar(self, y):
        if y < 0:
            raise ValueError("Entropy must be greater than 0")
        if y > 1:
            raise ValueError("Entropy must be less than 1")
        # Get the closest entropy in the hash
        closest_entropy = None
        for key_min, key_max in zip(
            list(self.approx_inverse_entropy_hash.keys())[:-1],
            list(self.approx_inverse_entropy_hash.keys())[1:],
        ):
            if key_min <= y and y <= key_max:
                if y - key_min < key_max - y:
                    closest_entropy = key_min
                else:
                    closest_entropy = key_max
                break
        if closest_entropy is None:
            print(y)
            raise ValueError("Entropy not found")
        return self.approx_inverse_entropy_hash[closest_entropy]

    def approx_inverse_entropy(
        self, y: Union[torch.Tensor, np.ndarray, list, float]
    ) -> Union[torch.Tensor, np.ndarray, list, float]:
        if isinstance(y, torch.Tensor):
            return torch.tensor(
                [self.approx_inverse_entropy_scalar(item) for item in y],
                device=y.device,
            )

        elif isinstance(y, np.ndarray):
            return np.array([self.approx_inverse_entropy_scalar(item) for item in y])
        elif isinstance(y, list):
            return [self.approx_inverse_entropy_scalar(item) for item in y]
        elif isinstance(y, float):
            return self.approx_inverse_entropy_scalar(y)
        else:
            raise ValueError("Unknown type")

    def build_new_label_distribution(
        self, labels, target_entropy, generator=None, device=None
    ):
        gamma = self.approx_inverse_entropy(target_entropy)
        new_labels = torch.where(
            torch.rand(gamma.shape, generator=generator, device=gamma.device)
            < 1 - gamma,
            labels.argmax(-1),
            randint_like_except(gamma, 0, self.N, labels.argmax(-1)),
        )
        new_labels = new_labels.to(torch.long)
        new_labels = (
            torch.nn.functional.one_hot(new_labels, num_classes=self.N)
            .squeeze(0)
            .float()
        )
        return new_labels


def leaky_relu_entropy_repartion(x, threshold=0.5, entropy_at_threshold=0.2):
    return torch.where(
        x < threshold,
        x * entropy_at_threshold / threshold,
        1 + (x - 1) * (1 - entropy_at_threshold) / (1 - threshold),
    )


def randint_like_except(x, low, high, except_value, generator=None):
    temp = torch.randint(low, high - 1, x.shape, generator=generator, device=x.device)
    return torch.where(temp >= except_value, temp + 1, temp)
