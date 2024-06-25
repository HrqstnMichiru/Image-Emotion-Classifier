from torch import nn
import torch
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from train import inference,emotionModel
from pathlib import Path
from utils import plot_middle,load_metric,increment_path
import matplotlib.pyplot as plt 
import cv2

plt.rcParams["font.sans-serif"] = "MicroSoft YaHei"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (8, 6)


random.seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = str(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
path = Path("./image/detect/exp")
path, n = increment_path(path)


class args:
    pass

param = args()
param.batch_size = 32
param.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
metric=load_metric(8,param)
trans = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

test_path = "./test_image"
test_data = ImageFolder(root=test_path, transform=trans)
classes=test_data.classes
classes=np.array(classes)
test_loader = DataLoader(
    dataset=test_data,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    drop_last=False,
)
model = emotionModel(8,is_init=True)
best_pt = torch.load("./checkpoints/exp1/best.pt", map_location="cuda:0")
best_weights = best_pt["weights"]
optim_weights = best_pt["optimizer"]
model.load_state_dict(best_weights,False)
model=model.cuda()
criterion = nn.CrossEntropyLoss()
criterion.load_state_dict(optim_weights,False)
criterion=criterion.cuda()
data,label=next(iter(test_loader))
data=data.cuda()
with torch.no_grad():
    model.eval()
    pred=model(data).argmax(dim=-1)
    cls_idx=classes[pred.cpu()]
plt.figure(figsize=(8,6),dpi=100)
for i,img in enumerate(data):
    img=np.transpose(img.cpu().numpy(), (1, 2, 0))
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.subplot(2,4,i+1)
    plt.title(f"True:{classes[label[i]]}")
    plt.imshow(img)
    plt.text(
   25,250, f"Pred:{cls_idx[i]}", fontsize=12, color="black"
    )
    plt.axis("off")
plt.savefig('prediction.jpg')


test_accuracy, test_loss, evaluation,metric = inference(
    model, test_loader, criterion, param, metric
)
print(test_accuracy, test_loss)
acc = round(evaluation["acc"].cpu().clone().detach().item(), 3)
prec = round(evaluation["prec"].cpu().clone().detach().item(), 3)
rec = round(evaluation["rec"].cpu().clone().detach().item(), 3)
f1score = round(evaluation["f1score"].cpu().clone().detach().item(), 3)
plot_middle(path,metric,test_data.classes,1)
print(f"acc:{acc},prec:{prec},rec:{rec},f1score:{f1score}")
