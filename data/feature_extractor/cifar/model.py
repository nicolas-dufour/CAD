import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
import argparse
from torchvision.models.resnet import BasicBlock, _resnet


class CIFAR10Module(pl.LightningModule):
    def __init__(self, label_smoothing=0.1, label_noise=0.0):
        super().__init__()
        self.model = _resnet(BasicBlock, [1, 1, 1, 1], None, True, num_classes=10)

        self.model.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.model.maxpool = torch.nn.Identity()
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.label_noise = label_noise

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        self.label_noise_generator = torch.Generator(device=self.device).manual_seed(
            3407
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        if self.label_noise > 0:
            y_noise = torch.randint(0, 10, y.shape, device=y.device, generator=self.label_noise_generator)
            y = torch.where(
                torch.rand(y.shape, device=y.device, generator=self.label_noise_generator) < self.label_noise, y_noise, y
            )

        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.train_acc(y_hat, y)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(y_hat, y)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        return [optimizer], [scheduler]


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, dataset_fraction=1.0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_fraction = dataset_fraction

    def setup(self, stage=None):
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        self.train_dataset = CIFAR10(
            root="../../../datasets",
            train=True,
            download=True,
            transform=transform_train,
        )
        indices = torch.randperm(len(self.train_dataset))[
            : int(len(self.train_dataset) * self.dataset_fraction)
        ]
        self.train_dataset = torch.utils.data.Subset(
            self.train_dataset, indices.tolist()
        )
        self.val_dataset = CIFAR10(
            root="../../../datasets",
            train=False,
            download=True,
            transform=transform_test,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def main(dataset_fraction=1.0, label_smoothing=0.1, label_noise=0.0):
    # Init DataModule
    dm = CIFAR10DataModule(
        batch_size=128,
        num_workers=16,
        dataset_fraction=dataset_fraction,
    )
    # Init model from datamodule
    model = CIFAR10Module(label_smoothing=label_smoothing, label_noise=label_noise)

    checkpoints_callbacks = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="epoch{epoch:02d}-" + f"fraction_{dataset_fraction}",
        save_top_k=0,
        save_last=True,
        every_n_epochs=50,
    )
    checkpoints_callbacks.CHECKPOINT_NAME_LAST = f"last_{label_noise}"

    callbacks = [checkpoints_callbacks]
    # Init Trainer
    epochs = 200
    trainer = pl.Trainer(
        max_epochs=epochs,
        devices=1,
        callbacks=callbacks,
        logger=False,
        check_val_every_n_epoch=50,
    )

    # Train the model
    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-fraction", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--label-noise", type=float, default=0.0)
    args = parser.parse_args()
    main(args.dataset_fraction, args.label_smoothing, args.label_noise)

# Accuracies
# Train/Val
# 75%: 0.900/ 0.843
# 50%: 0.794 / 0.901
# 25%: 0.899 / 0.661
