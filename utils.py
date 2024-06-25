# -*- coding: UTF-8 -*-
from pathlib import Path
import os
import random
import torch
import numpy as np
from prettytable import PrettyTable as pt
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    MulticlassConfusionMatrix,
    MulticlassPrecisionRecallCurve,
    MulticlassROC,
)
import matplotlib.pyplot as plt
from torchvision import transforms


def increment_path(path):
    path = Path(path)
    for n in range(1, 9999):
        p = f"{path}{n}"
        if not os.path.exists(p):
            break
    path = Path(p)
    path.mkdir(exist_ok=True)
    return path, n


def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enable = True


def check_GPU():
    table = pt()
    table.field_names = ["GPU是否可用", "GPU数量", "CUDA版本", "GPU索引号", "GPU名称"]
    isGPU = torch.cuda.is_available()
    table.add_row(
        [
            isGPU,
            torch.cuda.device_count(),
            torch.version.cuda,
            torch.cuda.current_device(),
            torch.cuda.get_device_name(0),
        ]
    )
    print(table)
    return isGPU


def load_metric(num_class, opt):
    metric = MetricCollection(
        {
            "acc": Accuracy(
                task="multiclass", num_classes=num_class, average="macro"
            ).to(opt.device),
            "prec": Precision(
                task="multiclass", num_classes=num_class, average="macro"
            ).to(opt.device),
            "rec": Recall(task="multiclass", num_classes=num_class, average="macro").to(
                opt.device
            ),
            "f1score": F1Score(
                task="multiclass", num_classes=num_class, average="macro"
            ).to(opt.device),
            "CM": MulticlassConfusionMatrix(num_classes=num_class).to(opt.device),
            "PR": MulticlassPrecisionRecallCurve(num_classes=num_class).to(opt.device),
            "ROC": MulticlassROC(num_classes=num_class).to(opt.device),
        }
    )
    return metric


def plot_single(x, y, path, title):
    plt.plot(x, y)
    plt.title(title)
    plt.savefig(path)
    plt.close()


def plot_curve(opt, evaluation, path):
    x = np.arange(opt.epochs)
    plt.style.use("classic")
    plot_single(x, evaluation["train_loss"], path / "train_loss.jpg", "train_loss")
    plot_single(
        x, evaluation["train_accuracy"], path / "train_acc.jpg", "train_accuracy"
    )
    plot_single(x, evaluation["train_rec"], path / "train_rec.jpg", "train_recall")
    plot_single(x, evaluation["train_prec"], path / "train_prec.jpg", "train_precision")
    plot_single(
        x, evaluation["train_f1score"], path / "train_f1score.jpg", "train_f1score"
    )
    plot_single(
        x, evaluation["val_accuracy"], path / "val_accuracy.jpg", "val_accuracy"
    )


def plot_middle(path, metric, classes, epoch):
    metric["CM"].plot(add_text=True, labels=classes)
    plt.title(f"ConfusionMatrix_{epoch}")
    plt.savefig(path / f"ConfusionMatrix_{epoch}.jpg")
    plt.close()
    metric["PR"].plot(score=True)
    plt.title(f"PRcurve_{epoch}")
    plt.savefig(path / f"PRcurve_{epoch}.jpg")
    plt.close()
    metric["ROC"].plot(score=True)
    plt.title(f"ROCcurve_{epoch}")
    plt.savefig(path / f"ROCcurve_{epoch}.jpg")
    plt.close()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, optimizer, lr_scheduler, pbar):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, lr_scheduler, pbar)
        elif score < self.best_score + self.delta:
            self.counter += 1
            pbar.write(f"EarlyStopping counter: {self.counter} out of {self.patience}\n")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, lr_scheduler, pbar)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, lr_scheduler, pbar=None):
        """Saves model when validation loss decrease."""
        if self.verbose:
            pbar.write(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\n"
            )
        best_pt = {
            "weights": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }
        torch.save(best_pt, self.save_path)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss


def init_trans():
    trans = {
        "test": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "train": transforms.Compose(
            [
                # transforms.RandomChoice(
                #     [
                #         transforms.RandomResizedCrop(224, antialias=True),
                #         transforms.Resize((224, 224)),
                #     ],
                #     p=[0.5, 0.5],
                # ),
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224,224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(25),
                transforms.RandomAffine(
                    degrees=(-25, 25), translate=(0.3, 0.3), scale=(0.8, 1.2), shear=15
                ),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.8, 1.2)),
                transforms.ColorJitter(brightness=(0.85,1.15),contrast=(0.15,1.15),saturation=(0.85,1.15),hue=(-0.1,0.1)),
                # transforms.AutoAugment(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
    return trans
