import matplotlib.pyplot as plt
import os
plt.rcParams["font.sans-serif"] = "MicroSoft YaHei"
image_list=os.listdir('./testImages_artphoto')
emotion_list=os.listdir('./train_image')
a=[]
for i in range(len(emotion_list)):
    path=os.path.join('./train_image',emotion_list[i])
    a.append(len(os.listdir(path)))
x=range(len(a))
plt.figure(figsize=(8,6))
for i in x:
    bar = plt.bar(emotion_list[i], a[i], 0.5, color="lightblue", label=emotion_list[i])
    height=bar[0].get_height()
    plt.annotate(f'{height}',xy=(bar[0].get_x()+bar[0].get_width()/2,height),xytext=(0,3),textcoords="offset points",ha='center',va='bottom')
plt.legend()
plt.xlabel("emotion")
plt.ylabel("number")
plt.title("图片情感数据集分析")
plt.grid(axis="y", linestyle="--")
plt.savefig("emotion_num2.jpg")
plt.show()
