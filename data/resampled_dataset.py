from torch.utils.data import Dataset
from utils.entropy import EntropyOneHotWithTemperature
import torch
from tqdm import tqdm
import torchvision
import copy
from pathlib import Path
from hydra.utils import instantiate


class LabelResampledDataset(Dataset):
    def __init__(
        self,
        dataset,
        dataset_name,
        num_classes,
        retrieve_gt=False,
        resampling_seed=3407,
        relabeling_batch_size=512,
        filter_threshold=0.0,
        bin_entropies=False,
        num_bins=10,
        uniform_bins=False,
        max_samples_for_bins=1e5,
        min_max_normalization=False,
        max_entropy_quantile=1.0,
        min_entropy_quantile=0.0,
    ):
        super().__init__()
        self._retrieve_gt = retrieve_gt
        self.dataset = dataset()
        self._transform = self.dataset.transform
        self.num_classes = num_classes
        self.relabeling_batch_size = relabeling_batch_size
        self.resampling_seed = resampling_seed
        self.dataset_name = dataset_name
        self.bin_entropies = bin_entropies
        self.min_max_normalization = min_max_normalization

        self.relabeled_dataset = self.relabel_dataset()
        print("Computing relabeling statistics")

        self.compute_relabeling_statistics()
        self.filtered_indices = [
            i
            for i, (_, entropy) in self.relabeled_dataset.items()
            if entropy > filter_threshold
        ]
        print(
            f"Filtered dataset size: {len(self.filtered_indices)} with threshold {filter_threshold}"
        )
        if self.bin_entropies or self.min_max_normalization:
            if len(self.filtered_indices) > max_samples_for_bins:
                entropies = [
                    self.relabeled_dataset[i.item()][1]
                    for i in torch.randperm(
                        len(self.filtered_indices),
                        generator=torch.Generator().manual_seed(resampling_seed),
                    )[: int(max_samples_for_bins)]
                ]

            else:
                entropies = [
                    self.relabeled_dataset[i][1] for i in self.filtered_indices
                ]
            print("Computing bins for entropy")
            if self.bin_entropies:
                if uniform_bins:
                    self.bins = torch.linspace(0, 1, num_bins + 1)
                else:
                    self.bins = self.compute_bins(
                        entropies,
                        num_bins,
                    )
            else:
                self.max_entropy_value, self.min_entropy_value = torch.quantile(
                    torch.tensor(entropies),
                    q=torch.tensor([max_entropy_quantile, min_entropy_quantile]),
                ).tolist()
                print(
                    f"Max entropy value: {self.max_entropy_value}, Min entropy value: {self.min_entropy_value}"
                )
        self.idx_mapping = {i: idx for i, idx in enumerate(self.filtered_indices)}

    def __len__(self):
        if not self.retrieve_gt:
            return len(self.filtered_indices)
        else:
            return len(self.dataset)

    def __getitem__(self, idx):
        if not self.retrieve_gt:
            idx = self.idx_mapping[idx]
            image = self.dataset[idx][0]
            new_label = self.relabeled_dataset[idx][0]
            target_entropy = self.relabeled_dataset[idx][1]
            if self.bin_entropies:
                target_entropy = (
                    torch.bucketize(torch.tensor(target_entropy), self.bins).item() - 1
                ) / (len(self.bins) - 2)
            elif self.min_max_normalization:
                if target_entropy > self.max_entropy_value:
                    target_entropy = 1.0
                elif target_entropy < self.min_entropy_value:
                    target_entropy = 0.0
                else:
                    target_entropy = (target_entropy - self.min_entropy_value) / (
                        self.max_entropy_value - self.min_entropy_value
                    )
            target_entropy = torch.tensor(target_entropy, dtype=torch.float32)
            return image, new_label, target_entropy
        else:
            image, label = self.dataset[idx]
            target_entropy = torch.tensor(1.0, dtype=torch.float32)
            return image, label, target_entropy.float()

    def compute_bins(self, entropies, num_bins):
        bins = torch.quantile(
            torch.tensor(entropies),
            q=torch.linspace(0, 1, num_bins + 1),
        )
        bins[0] = 0.0
        bins[-1] = 1.0
        return bins

    def compute_relabeling_statistics(self):
        acc = 0
        entropy_avg = 0
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.relabeling_batch_size,
            shuffle=False,
            num_workers=8,
        )
        for i, (images, true_labels) in enumerate(
            tqdm(dataloader, total=len(dataloader))
        ):
            batch_size = true_labels.shape[0]
            new_label = []
            for j in range(batch_size):
                new_label.append(self.relabeled_dataset[i * batch_size + j][0])
                entropy_avg += self.relabeled_dataset[i * batch_size + j][1]
            new_label = torch.stack(new_label)
            acc += (true_labels.argmax(dim=-1) == new_label.argmax(dim=-1)).sum().item()
        print(f"Dataset relabeling Accuracy: {acc / len(self.relabeled_dataset)}")
        print(f"Dataset total entropy: {entropy_avg / len(self.relabeled_dataset)}")
        del dataloader

    def relabel_dataset(self):
        device = self.relabel_setup()
        relabeled_dataset = {}
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.relabeling_batch_size,
            shuffle=False,
            num_workers=64,
        )
        with torch.no_grad():
            for i, (images, true_labels) in enumerate(
                tqdm(dataloader, total=len(dataloader))
            ):
                new_labels, target_entropies = self.relabel_batch(
                    images, true_labels, i, device
                )
                relabeled_dataset.update(
                    {
                        i * self.relabeling_batch_size
                        + j: (
                            new_labels[j],
                            target_entropies[j].item(),
                        )
                        for j in range(images.shape[0])
                    }
                )
        del dataloader
        assert len(relabeled_dataset) == len(self.dataset)
        self.relabel_teardown()
        return relabeled_dataset

    def relabel_setup(self):
        raise NotImplementedError

    def relabel_batch(self, images, true_labels, batch_idx, device):
        raise NotImplementedError

    def relabel_teardown(self):
        raise NotImplementedError

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        self._transform = transform
        self.dataset.transform = transform

    @property
    def retrieve_gt(self):
        return self._retrieve_gt

    @retrieve_gt.setter
    def retrieve_gt(self, retrieve_gt):
        self._retrieve_gt = retrieve_gt


class LeakyReLULabelResampledDataset(LabelResampledDataset):
    def __init__(
        self,
        dataset,
        dataset_name,
        num_classes,
        entropy_repartition_function,
        retrieve_gt=False,
        resampling_seed=3407,
        relabeling_batch_size=512,
        filter_threshold=0.0,
        bin_entropies=False,
        num_bins=10,
        uniform_bins=False,
        max_samples_for_bins=1e5,
        min_max_normalization=False,
        max_entropy_quantile=1.0,
        min_entropy_quantile=0.0,
    ):
        self.entropy_repartition_function = entropy_repartition_function
        super().__init__(
            dataset,
            dataset_name,
            num_classes,
            retrieve_gt=retrieve_gt,
            resampling_seed=resampling_seed,
            relabeling_batch_size=relabeling_batch_size,
            filter_threshold=filter_threshold,
            bin_entropies=bin_entropies,
            num_bins=num_bins,
            uniform_bins=uniform_bins,
            max_samples_for_bins=max_samples_for_bins,
            min_max_normalization=min_max_normalization,
            max_entropy_quantile=max_entropy_quantile,
            min_entropy_quantile=min_entropy_quantile,
        )

    def relabel_setup(self):
        device = torch.device("cpu")
        self.entropy_function = EntropyOneHotWithTemperature(N=self.num_classes)
        self.relabel_generator = torch.Generator(device).manual_seed(
            self.resampling_seed
        )
        return device

    def relabel_batch(self, images, true_labels, batch_idx, device):
        batch_size = true_labels.shape[0]
        true_labels = true_labels.to(device)
        target_entropies = self.entropy_repartition_function(
            torch.rand(
                batch_size,
                generator=self.relabel_generator,
                device=device,
            )
        )
        new_labels = self.entropy_function.build_new_label_distribution(
            true_labels,
            target_entropies,
            generator=self.relabel_generator,
            device=device,
        )
        return new_labels, 1 - target_entropies

    def relabel_teardown(self):
        del self.entropy_function
        del self.relabel_generator


class StructuredLabelResampledDataset(LabelResampledDataset):
    def __init__(
        self,
        dataset,
        dataset_name,
        num_classes,
        temperature: 1.0,
        retrieve_gt=False,
        resampling_seed=3407,
        relabeling_batch_size=512,
        filter_threshold=0.0,
        bin_entropies=False,
        num_bins=10,
        uniform_bins=False,
        max_samples_for_bins=1e5,
        min_max_normalization=False,
        max_entropy_quantile=1.0,
        min_entropy_quantile=0.0,
        store_logits_path=None,
    ):
        self.temperature = temperature
        self.store_logits_path = store_logits_path
        super().__init__(
            dataset,
            dataset_name,
            num_classes,
            retrieve_gt=retrieve_gt,
            resampling_seed=resampling_seed,
            relabeling_batch_size=relabeling_batch_size,
            filter_threshold=filter_threshold,
            bin_entropies=bin_entropies,
            num_bins=num_bins,
            uniform_bins=uniform_bins,
            max_samples_for_bins=max_samples_for_bins,
            min_max_normalization=min_max_normalization,
            max_entropy_quantile=max_entropy_quantile,
            min_entropy_quantile=min_entropy_quantile,
        )

    def relabel_setup(self):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.relabel_generator = torch.Generator().manual_seed(self.resampling_seed)
        self.max_entropy = torch.log(torch.tensor(self.num_classes))

        if (
            self.store_logits_path is not None
            and (
                Path(self.store_logits_path) / Path(f"{self.dataset_name}.pt")
            ).exists()
        ):
            self.logits = torch.load(
                Path(self.store_logits_path) / Path(f"{self.dataset_name}.pt")
            )
        else:
            self.logits = None
            Path(self.store_logits_path).mkdir(parents=True, exist_ok=True)
            self.all_logits = []
            if self.dataset_name.startswith("CIFAR-10"):
                import timm
                import torch.nn as nn

                self.num_classes = 10
                self.classifier_size = 32
                self.classifier = timm.create_model(
                    "hf_hub:edadaltocg/resnet18_cifar10", pretrained=False
                )
                self.classifier.conv1 = nn.Conv2d(
                    3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                )
                self.classifier.maxpool = nn.Identity()  # type: ignore
                self.classifier.fc = nn.Linear(512, 10)

                self.classifier.load_state_dict(
                    torch.hub.load_state_dict_from_url(
                        "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin",
                        map_location="cpu",
                        file_name="resnet18_cifar10.pth",
                    )
                )
                self.classifier = self.classifier.to(device)
                self.classifier.eval()
                self.transforms = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize(
                            (self.classifier_size, self.classifier_size), antialias=True
                        ),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                            mean=0.5,
                            std=0.5,
                        ),
                    ]
                )
            elif self.dataset_name.startswith("ImageNet"):
                from transformers import ViTForImageClassification

                self.num_classes = 1000
                self.classifier = ViTForImageClassification.from_pretrained(
                    "facebook/deit-small-patch16-224"
                ).to(device)

                self.transforms = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize(
                            (224, 224),
                            antialias=True,
                            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                        ),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                            mean=0.5,
                            std=0.5,
                        ),
                    ],
                )
            else:
                raise ValueError("Dataset not supported")
            self.classifier.eval()
            self.buffer_transform = copy.deepcopy(self.dataset.transform)
            self.dataset.transform = self.transforms
        return device

    def relabel_batch(self, images, true_labels, batch_idx, device):
        if self.logits is None:
            images = images.to(device)
            logits = self.classifier(images)
            if hasattr(logits, "logits"):
                logits = logits.logits.cpu()
            elif isinstance(logits, torch.Tensor):
                logits = logits.cpu()
            else:
                raise ValueError("Logits not supported")
            self.all_logits.append(logits)
        else:
            batch_size = true_labels.shape[0]
            logits = self.logits[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        new_labels_dist = (logits / self.temperature).softmax(dim=-1)
        new_labels = torch.multinomial(
            new_labels_dist,
            num_samples=1,
            replacement=False,
            generator=self.relabel_generator,
        ).squeeze(1)
        new_labels = torch.nn.functional.one_hot(
            new_labels, num_classes=self.num_classes
        )
        entropies = (
            -torch.sum(new_labels_dist * torch.log(new_labels_dist), dim=-1)
            / self.max_entropy
        )
        return new_labels, 1 - entropies

    def relabel_teardown(self):
        if self.logits is None:
            self.dataset.transform = self.buffer_transform
            del self.buffer_transform
            del self.relabel_generator
            del self.classifier
            del self.transforms
            self.logits = torch.cat(self.all_logits, dim=0)
            torch.save(
                self.logits,
                Path(self.store_logits_path) / Path(f"{self.dataset_name}.pt"),
            )
            del self.all_logits
        del self.logits
        del self.max_entropy
        torch.cuda.empty_cache()

    def relabel_dataset(self):
        device = self.relabel_setup()
        relabeled_dataset = {}
        if self.logits is None:
            dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.relabeling_batch_size,
                shuffle=False,
                num_workers=8,
            )
            print("Relabeling dataset")
            with torch.no_grad():
                for i, (images, true_labels) in enumerate(
                    tqdm(dataloader, total=len(dataloader))
                ):
                    new_labels, target_entropies = self.relabel_batch(
                        images, true_labels, i, device
                    )
                    relabeled_dataset.update(
                        {
                            i * self.relabeling_batch_size
                            + j: (
                                new_labels[j],
                                target_entropies[j].item(),
                            )
                            for j in range(images.shape[0])
                        }
                    )
            del dataloader
        else:
            for i in range(
                len(self.dataset) // self.relabeling_batch_size
                + (len(self.dataset) % self.relabeling_batch_size > 0)
            ):
                batch_size = (
                    self.relabeling_batch_size
                    if i < len(self.dataset) // self.relabeling_batch_size
                    else len(self.dataset) % self.relabeling_batch_size
                )
                new_labels, target_entropies = self.relabel_batch(
                    torch.zeros(batch_size), torch.zeros(batch_size), i, device
                )
                relabeled_dataset.update(
                    {
                        i * self.relabeling_batch_size
                        + j: (
                            new_labels[j],
                            target_entropies[j].item(),
                        )
                        for j in range(batch_size)
                    }
                )
        assert len(relabeled_dataset) == len(self.dataset)
        self.relabel_teardown()
        return relabeled_dataset


class SoftmaxLabelResampledDataset(LabelResampledDataset):
    def __init__(
        self,
        dataset,
        dataset_name,
        num_classes,
        resampling_model,
        retrieve_gt=False,
        resampling_seed=3407,
        relabeling_batch_size=512,
        filter_threshold=0.0,
        bin_entropies=False,
        num_bins=10,
        uniform_bins=False,
        max_samples_for_bins=1e5,
        min_max_normalization=False,
        max_entropy_quantile=1.0,
        min_entropy_quantile=0.0,
        store_logits_path=None,
    ):
        self.store_logits_path = store_logits_path
        self.resampling_model = resampling_model
        super().__init__(
            dataset,
            dataset_name,
            num_classes,
            retrieve_gt=retrieve_gt,
            resampling_seed=resampling_seed,
            relabeling_batch_size=relabeling_batch_size,
            filter_threshold=filter_threshold,
            bin_entropies=bin_entropies,
            num_bins=num_bins,
            uniform_bins=uniform_bins,
            max_samples_for_bins=max_samples_for_bins,
            min_max_normalization=min_max_normalization,
            max_entropy_quantile=max_entropy_quantile,
            min_entropy_quantile=min_entropy_quantile,
        )

    def relabel_setup(self):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if (
            self.store_logits_path is not None
            and (
                Path(self.store_logits_path) / Path(f"{self.dataset_name}.pt")
            ).exists()
        ):
            self.logits = torch.load(
                Path(self.store_logits_path) / Path(f"{self.dataset_name}.pt")
            )
        else:
            self.logits = None
            Path(self.store_logits_path).mkdir(parents=True, exist_ok=True)
            self.all_logits = []
            if self.dataset_name.startswith("CIFAR-10"):
                self.resampling_model = self.resampling_model.to(device)
                self.resampling_model.eval()
                self.transforms = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
                    ]
                )
            elif self.dataset_name.startswith("ImageNet"):
                raise NotImplementedError
            else:
                raise ValueError("Dataset not supported")
            self.buffer_transform = copy.deepcopy(self.dataset.transform)
            self.dataset.transform = self.transforms
        return device

    def relabel_batch(self, images, true_labels, batch_idx, device):
        if self.logits is None:
            images = images.to(device)
            logits = self.resampling_model(images)
            if hasattr(logits, "logits"):
                logits = logits.logits.cpu()
            elif isinstance(logits, torch.Tensor):
                logits = logits.cpu()
            else:
                raise ValueError("Logits not supported")
            self.all_logits.append(logits)
        else:
            batch_size = true_labels.shape[0]
            logits = self.logits[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        entropies, new_labels = logits.softmax(dim=-1).max(dim=-1)
        new_labels = torch.nn.functional.one_hot(
            new_labels, num_classes=self.num_classes
        ).float()
        return new_labels, entropies

    def relabel_teardown(self):
        if self.logits is None:
            self.dataset.transform = self.buffer_transform
            del self.buffer_transform
            del self.resampling_model
            del self.transforms
            self.logits = torch.cat(self.all_logits, dim=0)
            torch.save(
                self.logits,
                Path(self.store_logits_path) / Path(f"{self.dataset_name}.pt"),
            )
            del self.all_logits
        del self.logits
        torch.cuda.empty_cache()

    def relabel_dataset(self):
        device = self.relabel_setup()
        relabeled_dataset = {}
        if self.logits is None:
            dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.relabeling_batch_size,
                shuffle=False,
                num_workers=8,
            )
            with torch.no_grad():
                for i, (images, true_labels) in enumerate(
                    tqdm(dataloader, total=len(dataloader))
                ):
                    new_labels, target_entropies = self.relabel_batch(
                        images, true_labels, i, device
                    )
                    relabeled_dataset.update(
                        {
                            i * self.relabeling_batch_size
                            + j: (
                                new_labels[j],
                                target_entropies[j].item(),
                            )
                            for j in range(images.shape[0])
                        }
                    )
            del dataloader
        else:
            for i in range(
                len(self.dataset) // self.relabeling_batch_size
                + (len(self.dataset) % self.relabeling_batch_size > 0)
            ):
                batch_size = (
                    self.relabeling_batch_size
                    if i < len(self.dataset) // self.relabeling_batch_size
                    else len(self.dataset) % self.relabeling_batch_size
                )
                new_labels, target_entropies = self.relabel_batch(
                    torch.zeros(batch_size), torch.zeros(batch_size), i, device
                )
                relabeled_dataset.update(
                    {
                        i * self.relabeling_batch_size
                        + j: (
                            new_labels[j],
                            target_entropies[j].item(),
                        )
                        for j in range(batch_size)
                    }
                )
        assert len(relabeled_dataset) == len(self.dataset)
        self.relabel_teardown()
        return relabeled_dataset
