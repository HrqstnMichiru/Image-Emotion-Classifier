# -*- coding: UTF-8 -*-
import copy
import torch
import torch.nn as nn
import datetime
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,random_split,Dataset
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time
import argparse
import sys
from utils import EarlyStopping, load_metric, check_GPU, set_seed, increment_path, plot_middle,EarlyStopping,init_trans
import pickle
import timm
import warnings

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = "MicroSoft YaHei"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (8, 6)


class emotionModel(nn.Module):
    def __init__(self, num_class, is_init=False):
        super().__init__()
        # self.model = torch.hub.load(
        #     "NVIDIA/DeepLearningExamples:torchhub", "nvidia_resnet50", pretrained=True
        # )
        # self.model = timm.create_model(
        #     "resnet34.bt_in1k",
        #     num_classes=num_class,
        #     pretrained=True,
        #     pretrained_cfg_overlay=dict(
        #         file="E:/huggingface_cache/hub/models--timm--resnet34.bt_in1k/snapshots/f88f48c768940f9894e96ed81b58c7ce2eaf2c0a/pytorch_model.bin"
        #     )
        # )
        # self.model = timm.create_model(
        #     "vit_small_patch16_224",
        #     pretrained=False,
        #     checkpoint_path="E:/huggingface_cache/hub/models--timm--vit_small_patch16_224.augreg_in1k/snapshots/9f817e4a98515abc212f00cd836b960c556029d8/model.safetensors",
        # )
        # self.model.head = nn.Linear(self.model.head.in_features, num_class)
        # self.model=torchvision.models.resnet34()
        # self.model=torchvision.models.resnet50()
        self.model = timm.create_model(
            "resnet18.a3_in1k",
            pretrained=False,
            checkpoint_path="E:/huggingface_cache/hub/models--timm--resnet18.a3_in1k/snapshots/3edbf22272fbbca56c25be1d4f1f3055bed72ee0/model.safetensors",
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_class)
        if not is_init:
            self.model.fc.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            nn.init.constant_(layer.bias, 0.0)
        if isinstance(layer, nn.BatchNorm2d):
            nn.init.normal_(layer.weight, 1.0, 0.02)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, input):
        output = self.model(input)
        return output

class AugmentedDataset(Dataset):
    def __init__(self,data,trans,num_augmented=3):
        super().__init__()
        self.original_data = data
        self.trans = trans
        self.num_augmented = num_augmented
        self.augmented_data=[]
        self.labels=[]
        self.imgAugment()

    def imgAugment(self):
        pbar = tqdm(self.original_data, leave=True, desc="正在进行数据增强")
        for img, label in pbar:
            imgs = [self.trans["train"](img) for _ in range(self.num_augmented)]
            self.augmented_data.append(self.trans["test"](img))
            self.augmented_data.extend(imgs)
            labels = [label for _ in range(self.num_augmented + 1)]
            self.labels.extend(labels)
        self.labels=torch.LongTensor(self.labels)

    def __len__(self):
        return len(self.original_data) * (self.num_augmented+1)
    
    def __getitem__(self, index):
        return self.augmented_data[index],self.labels[index]


class ValidationDataset(Dataset):
    def __init__(self,data,trans):
        super().__init__()
        self.data=data
        self.trans=trans

    def __getitem__(self, index):
        return self.trans(self.data[index][0]), self.data[index][1]

    def __len__(self):
        return len(self.data)


def init_config(opt):
    trans=init_trans()
    train_path = "./train_image"
    train_data_ = ImageFolder(root=train_path)
    train_size = int(opt.train_ratio * len(train_data_))
    val_size = len(train_data_) - train_size
    test_path = "./test_image"
    test_data = ImageFolder(root=test_path, transform=trans["test"])
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=32,
        shuffle=True,
        drop_last=False,
    )
    train_data, val_data = random_split(train_data_, [train_size, val_size])
    train_data=train_data_
    # print(len(train_data))
    # print(len(val_data))
    val_data=ValidationDataset(val_data,trans["test"])
    if opt.augmented:
        if opt.re_augmented:
            train_data=AugmentedDataset(train_data,trans,opt.num_augment)
            with open('augumented_data2.pkl','wb+') as f:
                pickle.dump(train_data,f)
            # with open('val_data.pkl','wb+') as f:
            #     pickle.dump(val_data, f)
        else:
            with open('augumented_data2.pkl','rb+') as f:
                train_data=pickle.load(f)
            # with open("val_data.pkl", "rb+") as f:
            #     val_data=pickle.load(f)
    else:
        train_data = ValidationDataset(train_data,trans["test"])
    train_loader = DataLoader(
        dataset=train_data, batch_size=opt.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        dataset=val_data, batch_size=opt.batch_size, shuffle=True, drop_last=False
    )
    metric = load_metric(8, opt)
    args = {
        "train_loader": train_loader,
        # "val_loader": val_loader,
        "test_loader": test_loader,
        "class2idx": train_data_.class_to_idx,
        "classes": train_data_.classes,
        "metric": metric,
    }
    return args


@torch.no_grad()
def inference(model, loader, criterion, opt, metric, epoch=0, isVal=False):
    loss = 0
    accuracy = 0
    model.eval()
    if isVal:
        pbar = tqdm(loader, desc=f"Validation,Epoch {epoch}", leave=True)
    else:
        pbar = tqdm(loader, desc=f"Testing", leave=True)
    for data in pbar:
        imgs, labels = data
        imgs = imgs.to(opt.device)
        labels = labels.to(opt.device)
        output = model(imgs)
        loss += criterion(output, labels).cpu().clone().detach().item()
        temp = (
            (output.argmax(axis=-1)) == labels
        ).sum().cpu().clone().detach().item() / opt.batch_size
        accuracy += temp
        metric.forward(output, labels)
        pbar.set_postfix(batch_acc=temp)
    evaluation = metric.compute()
    return accuracy / len(loader), loss / len(loader), evaluation, metric


def train( opt, args):
    if not opt.early_stop:
        best_acc=0
    model = emotionModel(8)
    train_loader = args["train_loader"]
    # val_loader = args["val_loader"]
    test_loader = args["test_loader"]
    metric = args["metric"]
    classes = args["classes"]
    optimizer = optim.Adam(
        model.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay=opt.weight_decay
    )
    # optimizer=optim.SGD(model.parameters(),lr=opt.lr,momentum=0.9,weight_decay=opt.weight_decay,nesterov=True)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt.epochs * len(train_loader), eta_min=opt.eta_min
    )
    if opt.resume:
        best_pt = torch.load("./checkpoints/exp1/best.pt", map_location="cuda:0")
        best_weights = best_pt["weights"]
        optim_weights = best_pt["optimizer"]
        lr_weights = best_pt["lr_scheduler"]
        model.load_state_dict(best_weights, False)
        criterion.load_state_dict(optim_weights,False)
        lr_scheduler.load_state_dict(lr_weights)
    model=model.to(opt.device)
    criterion=criterion.to(opt.device)
    if opt.early_stop:
        early_stopping = EarlyStopping(opt.path_pt/"best.pt",verbose=True,patience=7,delta=0.01)
    for epoch in range(opt.epochs):
        epoch += 1
        train_loss = 0
        pbar = tqdm(train_loader, leave=True, colour="red")
        start = time.time()
        model.train()
        for data in pbar:
            pbar.set_description(f"Training,Epoch [{epoch}/{opt.epochs}]")
            optimizer.zero_grad()
            imgs, labels = data
            imgs = imgs.to(opt.device)
            labels = labels.to(opt.device)
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_loss += loss.cpu().clone().detach().item()
            batch_metrics = metric.forward(output, labels)
            pbar.set_postfix(
                batch_loss=loss.item(), batch_acc=batch_metrics["acc"].item()
            )
        end = time.time()
        train_time = round(end - start, 3)
        train_loss = train_loss / len(train_loader)
        train_metrics = metric.compute()
        acc = round(train_metrics["acc"].cpu().clone().detach().item(), 3)
        prec = round(train_metrics["prec"].cpu().clone().detach().item(), 3)
        rec = round(train_metrics["rec"].cpu().clone().detach().item(), 3)
        f1score = round(train_metrics["f1score"].cpu().clone().detach().item(), 3)
        if epoch % 10 == 0:
            root = opt.root / f"epoch{epoch}"
            if not root.exists():
                root.mkdir()
            plot_middle(root, metric, classes, epoch)
        metric.reset()
        val_accuracy, val_loss, evaluation, metric = inference(
            model, test_loader, criterion, opt, metric, epoch, isVal=True
        )
        metric.reset()
        val_acc = evaluation["acc"].cpu().clone().detach().item()
        val_rec = evaluation["rec"].cpu().clone().detach().item()
        val_prec = evaluation["prec"].cpu().clone().detach().item()
        val_f1score = evaluation["f1score"].cpu().clone().detach().item()
        pbar.write(
            f"Time:{train_time}s,Loss:{train_loss},Acc:{acc},Prec:{prec},Rec:{rec},f1score:{f1score}"
        )
        pbar.write(f"The acc of validation:{val_accuracy}")
        pbar.write(f"The loss of validation:{val_loss}")
        writer.add_scalars(
            "accuracy", {"train": acc, "val": val_accuracy, "val_auto": val_acc}, epoch
        )
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("recall", {"train": rec, "val": val_rec}, epoch)
        writer.add_scalars("precision", {"train": prec, "val": val_prec}, epoch)
        writer.add_scalars("f1score", {"train": f1score, "val": val_f1score}, epoch)
        writer.flush()
        if opt.early_stop:
            early_stopping(val_loss, model, optimizer, lr_scheduler,pbar)
        else:
            if val_accuracy > best_acc:
                best_acc = val_accuracy
                best_weights = copy.deepcopy(model.state_dict())
                best_pt = {
                    "epoch": epoch,
                    "weights": best_weights,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                }
                torch.save(best_pt, opt.path_pt / "best.pt")
        if early_stopping.early_stop:
            pbar.write("Early stopping")
            break
        torch.save(model.state_dict(), opt.path_pt / "last.pt")
    writer.add_graph(model, input_to_model=torch.randn(1, 3, 224, 224).to(opt.device))
    writer.close()


def parse_opt(known=False):
    ret = check_GPU()
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="total training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="total batch size")
    parser.add_argument(
        "--device",
        type=str,
        default=torch.device("cuda:0" if ret else "cpu"),
        help="device",
    )
    parser.add_argument("-l", "--lr", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument("--eta_min", type=float, default=0, help="the Min Value Of Lr_scheduler")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training Dataset Ratio")
    parser.add_argument("--name", type=str, default="exp", help="Save Basename")
    parser.add_argument("--early_stop",action="store_false", help="Is earlyStopping?")#?store貌似是相反的,你不输入就相反,输入才相同
    parser.add_argument("--resume", action="store_true", help="Is resume?")
    parser.add_argument("--num_augment", type=int, default=10, help="the number of augment")
    parser.add_argument("--re_augmented", action="store_true",help="Is re:augmented")
    parser.add_argument("--augmented", action="store_true", help="Is augmented")
    parser.add_argument("--weight_decay",type=float,default=2e-3,help='regularization')
    return parser.parse_known_args()[0] if known else parser.parse_args()

def main(opt):
    args = init_config(opt)
    start_time = datetime.datetime.now()
    train(opt, args)
    end_time = datetime.datetime.now()
    print(f"Training has completed for {end_time-start_time}")


if __name__ == "__main__":
    opt = parse_opt(False)
    kerasPath, _ = increment_path("./log/log")
    root, _ = increment_path("./image/train/exp")
    opt.root = root
    path_pt, _ = increment_path("./checkpoints/exp")
    opt.path_pt = path_pt
    global writer
    writer = SummaryWriter(
        log_dir=kerasPath, comment="这是一个有趣的邂逅", flush_secs=120
    )
    set_seed(42)
    sys.exit(main(opt))
