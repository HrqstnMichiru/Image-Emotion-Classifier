import os
import shutil
from pathlib import Path
import tqdm
import imageio.v2 as ia
import imgaug
from imgaug import augmenters as iaa

imgaug.seed(42)
sometimes=lambda aug:iaa.Sometimes(0.5,aug)#以p=0.5的概率去执行sometimes传递的图像增强

seq1 = iaa.Sequential(
    [
        iaa.Resize(224),
        iaa.Flipud(0.2),
        iaa.Fliplr(0.5),
        # 随机裁剪图片边长比例的0~0.1
        iaa.Crop(percent=(0, 0.1)),
        # Sometimes是指指针对50%的图片做处理
        iaa.Sometimes(
            0.5,
            # 高斯模糊
            iaa.GaussianBlur(sigma=(0, 0.5)),
        ),
        # 增强或减弱图片的对比度
        iaa.LinearContrast((0.75, 1.25)),
        # 添加高斯噪声
        # 对于50%的图片,这个噪采样对于每个像素点指整张图片采用同一个值
        # 剩下的50%的图片，对于通道进行采样(一张图片会有多个值)
        # 改变像素点的颜色(不仅仅是亮度)
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # 让一些图片变的更亮,一些图片变得更暗
        # 对20%的图片,针对通道进行处理
        # 剩下的图片,针对图片进行处理
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # 仿射变换
        iaa.Affine(
            # 缩放变换
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            # 平移变换
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            # 旋转
            rotate=(-25, 25),
            # 剪切
            shear=(-8, 8),
        ),
        # 使用随机组合上面的数据增强来处理图片
    ],
    random_order=True,
)

seq2 = iaa.Sequential(
    [
        iaa.Resize(224),  # 改变图片大小
        iaa.Fliplr(0.5),  # 对50%的图像进行镜像翻转
        iaa.Flipud(0.2),  # 对20%的图像做左右翻转
        # 对随机的一部分图像做crop操作，crop的幅度为0到10%
        sometimes(iaa.Crop(percent=(0, 0.1), keep_size=True)),
        # 对一部分图像做仿射变换
        sometimes(
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # 图像缩放为80%到120%之间
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # 平移±20%之间
                rotate=(-45, 45),  # 旋转±45度之间
                shear=(-16, 16),  # 剪切变换±16度，（矩形变平行四边形）
                order=[0, 1],  # 使用最邻近差值或者双线性差值
                cval=(0, 255),  # 全白全黑填充
                mode=imgaug.ALL,  # 定义填充图像外区域的方法
            )
        ),
        # 使用下面的0个到5个之间的方法去增强图像
        iaa.SomeOf(
            (0, 5),
            [
                # 将部分图像进行超像素的表示
                sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                # 将部分图像进行高斯模糊，均值模糊，中值模糊中的一种
                iaa.OneOf(
                    [
                        iaa.GaussianBlur((0, 3.0)),
                        iaa.AverageBlur(k=(2, 7)),
                        iaa.MedianBlur(k=(3, 11)),
                    ]
                ),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # 锐化处理
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # 浮雕效果
                # 边缘检测，将检测到的赋值0或者255然后叠在原图上
                sometimes(
                    iaa.OneOf(
                        [
                            iaa.EdgeDetect(alpha=(0, 0.7)),
                            iaa.DirectedEdgeDetect(
                                alpha=(0, 0.7), direction=(0.0, 1.0)
                            ),
                        ]
                    )
                ),
                # 加入高斯噪声
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                ),
                # 将1%到10%的像素设置为黑色
                # 或者将3%到15%的像素用原图大小2%到5%的黑色方块覆盖
                iaa.OneOf(
                    [
                        iaa.Dropout((0.01, 0.1), per_channel=0.5),
                        iaa.CoarseDropout(
                            (0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2
                        ),
                    ]
                ),
                # 5%的概率反转像素的强度，即原来的强度为v那么现在的就是255-v
                iaa.Invert(0.05, per_channel=True),
                # 每个像素随机加减-10到10之间的数
                iaa.Add((-10, 10), per_channel=0.5),
                # 像素乘上0.5或者1.5之间的数字
                iaa.Multiply((0.5, 1.5), per_channel=0.5),
                # 将整个图像的对比度变为原来的一半或者二倍
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                # 将RGB变成灰度图然后乘alpha加在原图上
                iaa.Grayscale(alpha=(0.0, 1.0)),
                # 把像素移动到周围的地方
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                # 扭曲图像的局部区域
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
            ],
            random_order=True,
        ),
    ],
    random_order=True,
)


train_image_file = Path("./train_image")
test_image_file = Path("./test_image")
if not train_image_file.exists():
    train_image_file.mkdir()
if not test_image_file.exists():
    test_image_file.mkdir()
train_file = Path("testImages_artphoto_trainset.txt")
train_list = train_file.read_text().splitlines()
test_file = Path("testImages_artphoto_testset.txt")
test_list = test_file.read_text().splitlines()
print(len(test_list))
print(len(train_list))
print(len(os.listdir('./testImages_artphoto')))
image_src = Path("./testImages_artphoto")
image_list = os.listdir(image_src)
pbar = tqdm.tqdm(image_list, leave=True, desc="开始分割数据集!", position=0,ncols=80)
try:
    for image in pbar:
        image = image.strip()
        emotion = image.split("_")[0]
        if image in test_list:
            path = test_image_file.joinpath(emotion)
            if not path.exists():
                path.mkdir()
            shutil.copy(Path(image_src / image), Path(path / image))
        if image in train_list:
            path = train_image_file.joinpath(emotion)
            if not path.exists():
                path.mkdir()
            path_1 = Path(image_src / image)
            path_2 = Path(path / image)
            img = ia.imread(path_1)
            if len(img.shape) == 3:
                shutil.copy(path_1, path_2)
                [
                    ia.imwrite(str(path_2).split(".")[0] + f"_{i+1}" + ".jpg", seq1(image=img))
                    for i in range(15)
                ]
            else:
                shutil.copy(path_1, path_2)
except Exception as e:
    print(e)
else:
    print('你成功了')
finally:
    print('程序运行结束')
