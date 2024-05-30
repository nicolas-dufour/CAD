from .inception_metrics import MultiInceptionMetrics
from .classifier_acc import ClassifierAccuracy
from tqdm import tqdm
import torch
import torch.nn.functional as F
from copy import deepcopy
from torchmetrics.multimodal.clip_score import CLIPScore
import webdataset as wds
import random
import transformers
import string
from utils.image_processing import remap_image_torch


class SampleAndEval:
    def __init__(
        self,
        logger,
        num_images=50000,
        compute_conditional=True,
        compute_unconditional=False,
        compute_per_class_metrics=True,
        log_prefix="val",
        num_classes=10,
        eval_set="val",
        dataset_name="CIFAR-10",
        dataset_type="class_conditional",
        cfg_rate=0.0,
        shape=(4, 32, 32),
        negative_prompts=None,
        confidence_value=1.0,
        use_clip_for_fid=False,
    ):
        if use_clip_for_fid:
            feature = CLIPFeatureExtractor(path="openai/clip-vit-large-patch14")
        else:
            feature = 2048
        super().__init__()
        self.inception_metrics = MultiInceptionMetrics(
            feature,
            reset_real_features=False,
            compute_unconditional_metrics=compute_unconditional,
            compute_conditional_metrics=compute_conditional,
            compute_conditional_metrics_per_class=compute_per_class_metrics,
            num_classes=num_classes,
            num_inception_chunks=10,
            manifold_k=3,
        )
        if dataset_type == "class_conditional":
            self.classifier_acc = ClassifierAccuracy(dataset=dataset_name)
        elif dataset_type == "text_conditional":
            self.clip_score = CLIPScore("openai/clip-vit-large-patch14")
        else:
            raise ValueError(f"Unknown dataset_type {dataset_type}")

        self.dataset_type = dataset_type
        self.logger = logger
        self.num_images = num_images
        self.compute_conditional = compute_conditional
        self.compute_unconditional = compute_unconditional
        self.log_prefix = log_prefix
        self.true_features_computed = False
        self.eval_set = eval_set
        self.cfg_rate = cfg_rate
        self.num_classes = num_classes
        self.cfg_rate = cfg_rate
        self.shape = shape
        self.negative_prompts = negative_prompts
        self.confidence_value = confidence_value

    def compute_and_log_metrics(self, device, pl_module, datamodule, step=0, epoch=0):
        with torch.no_grad():
            self.inception_metrics.to(device)
            if self.dataset_type == "class_conditional":
                self.classifier_acc.to(device)
            elif self.dataset_type == "text_conditional":
                self.clip_score.to(device)
            if self.eval_set == "val":
                dataloader = datamodule.val_dataloader()
            elif self.eval_set == "train":
                dataloader = datamodule.train_dataloader()
            else:
                raise ValueError(f"Unknown eval_set {self.eval_set}")
            if self.dataset_type == "class_conditional":
                dataloader.dataset.transform = datamodule.val_aug
            dataloader.dataset.retrieve_gt = True
            pl_module = pl_module.to(device)
            pl_module.eval()
            if (
                not self.true_features_computed
                or not self.inception_metrics.reset_real_features
            ):
                self.compute_true_images_features(dataloader)
                self.true_features_computed = True
            if self.compute_unconditional:
                self.compute_fake_images_features(
                    pl_module, dataloader, conditional=False
                )
            if self.compute_conditional:
                self.compute_fake_images_features(
                    pl_module, dataloader, conditional=True
                )

            metrics = self.inception_metrics.compute()
            metrics = {f"{self.log_prefix}/{k}": v for k, v in metrics.items()}
            metrics["trainer/global_step"] = step
            metrics["epoch"] = epoch
            if self.compute_conditional and self.dataset_type == "class_conditional":
                metrics[f"{self.log_prefix}/classifier_acc"] = (
                    self.classifier_acc.compute()
                )
            elif self.compute_conditional and self.dataset_type == "text_conditional":
                metrics[f"{self.log_prefix}/clip_score"] = self.clip_score.compute()
            print(metrics)
            self.logger.experiment.log(
                metrics,
            )

    def compute_true_images_features(self, dataloader):
        if isinstance(dataloader, wds.DataPipeline):
            dataset_length = dataloader.pipeline[0].dataset.num_samples
        elif isinstance(dataloader, torch.utils.data.DataLoader):
            dataset_length = len(dataloader.dataset)
        else:
            raise ValueError(f"Unknown dataloader type {type(dataloader)}")
        if dataset_length < self.num_images:
            max_images = dataset_length
        else:
            max_images = self.num_images

        print("Computing true images features")
        max_batch_size = 0
        for i, batch in enumerate(tqdm(dataloader)):
            images = batch["image"]
            cond = batch["label"] if "label" in batch else None
            max_batch_size = max(max_batch_size, images.shape[0])
            if i * max_batch_size >= max_images:
                break
            elif i * max_batch_size + images.shape[0] > max_images:
                images = images[: max_images - i * max_batch_size]
                cond = cond[: max_images - i * max_batch_size]
            self.inception_metrics.update(
                remap_image_torch(images).to(self.inception_metrics.device),
                cond,
                image_type="real",
            )

    def compute_fake_images_features(self, pl_module, dataloader, conditional=False):
        if isinstance(dataloader, wds.DataPipeline):
            dataset_length = dataloader.pipeline[0].dataset.num_samples
        elif isinstance(dataloader, torch.utils.data.DataLoader):
            dataset_length = len(dataloader.dataset)
        else:
            raise ValueError(f"Unknown dataloader type {type(dataloader)}")
        if dataset_length < self.num_images:
            max_images = dataset_length
        else:
            max_images = self.num_images

        print("Computing fake images features")
        max_batch_size = 0
        for i, batch in enumerate(tqdm(dataloader)):
            if self.dataset_type == "text_conditional":
                images = batch["image"]
                text = batch["text"]
                cond = text
            else:
                images = batch["image"]
                cond = batch["label"] if "label" in batch else None
            shape = self.shape
            batch_size = images.shape[0]
            max_batch_size = max(max_batch_size, batch_size)
            if i * max_batch_size >= max_images:
                break
            elif i * max_batch_size + batch_size > max_images:
                batch_size = max_images - i * max_batch_size
                cond = cond[:batch_size]
            if isinstance(cond, torch.Tensor):
                cond = cond.to(pl_module.device)
            with torch.no_grad():
                if conditional:
                    images = pl_module.sample(
                        batch_size,
                        shape,
                        cond=cond,
                        stage=self.log_prefix,
                        cfg=self.cfg_rate,
                        unconfident_prompt=(
                            generate_negative_prompts(batch_size)
                            if self.negative_prompts == "random_prompt"
                            else None
                        ),
                        confidence_value=self.confidence_value,
                    )
                    self.inception_metrics.update(
                        images, cond, image_type="conditional"
                    )
                    if self.dataset_type == "class_conditional":
                        self.classifier_acc.update(images, cond)
                    elif self.dataset_type == "text_conditional":
                        self.clip_score.update(images, text)
                else:
                    images_uncond = pl_module.sample(
                        batch_size,
                        shape,
                        stage=self.log_prefix,
                        cfg=self.cfg_rate,
                        unconfident_prompt=(
                            generate_negative_prompts(batch_size)
                            if self.negative_prompts == "random_prompt"
                            else None
                        ),
                        confidence_value=self.confidence_value,
                    )
                    self.inception_metrics.update(
                        images_uncond, image_type="unconditional"
                    )


def generate_random_string():
    words = []
    for _ in range(random.randint(3, 6)):
        word = "".join(
            random.choice(string.ascii_lowercase) for _ in range(random.randint(3, 7))
        )
        words.append(word)
    return " ".join(words)


def generate_negative_prompts(batch_size):
    return [generate_random_string() for _ in range(batch_size)]


class CLIPFeatureExtractor(torch.nn.Module):
    def __init__(self, path="openai/clip-vit-large-patch32"):
        super().__init__()
        self.processor = transformers.CLIPImageProcessor.from_pretrained(path)
        self.model = transformers.CLIPVisionModel.from_pretrained(path)

    def forward(self, images):
        images = self.processor(images=images, return_tensors="pt").to(
            self.model.device
        )
        features = self.model(**images).last_hidden_state[:, 0, :]
        return features, torch.randn(
            features.shape[0], 1000, dtype=torch.float, device=features.device
        )
