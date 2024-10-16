import math
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from dataloader import get_dataloader
from model import get_resnet50, CNN


def imshow(img):
    # 反归一化，将数据重新映射到0-1之间
    img = img / 2 + 0.5
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.show()


# 处理图像，将 tensor 转换为 numpy 格式，并标准化
def preprocess_for_cam(images):
    imgs = []
    for img in images:
        np_img = img.numpy()
        np_img = np.transpose(np_img, (1, 2, 0))  # 转换为 (H, W, C)
        imgs.append((np_img * 0.5 + 0.5))  # 逆归一化
    return imgs


if __name__ == '__main__':
    use_cnn = False
    batch_size = 20
    image_size = 224 if not use_cnn else 32
    trainloader, testloader = get_dataloader(batch_size, image_size)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    if use_cnn:
        model = CNN()
        model.load_state_dict(torch.load('./model/cnn_adam.pth'))
        model.eval()  # 设置为评估模式
        # 设置CNN的目标层
        target_layers = [model.conv2]
    else:
        # 加载 ResNet50 模型
        model = get_resnet50()
        model.load_state_dict(torch.load('./model/resnet_18epochs.pth'))
        model.eval()  # 设置为评估模式
        # 对于 Grad-CAM，定义需要的目标层
        target_layers = [model.layer4[-1]]  # ResNet50 的最后一层卷积层
    # 从 trainloader 获取一个 batch 的图像
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # 可视化原始图片
    imshow(torchvision.utils.make_grid(images))

    # 将图像发送到模型进行 Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

    # 预处理 batch 中的每个图像
    input_tensor = images  # 直接使用 CIFAR-10 的数据（已经是 Tensor 格式）

    # 获取 Grad-CAM
    grayscale_cams = cam(input_tensor=input_tensor)

    # 对每张图片生成 Grad-CAM 热图并叠加
    processed_imgs = preprocess_for_cam(images)
    # 假设 images 是你输入的图像 Tensor
    num_images = len(images)  # 获取输入图片的数量

    # 动态计算行和列的数量，让图片展示更加自适应
    n_cols = min(8, num_images)  # 每行最多显示8张图片
    n_rows = math.ceil(num_images / n_cols)  # 根据总图片数计算行数

    # 创建自适应的子图布局
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    # 将 axes 处理为一个扁平化的列表
    axes = axes.flatten() if num_images > 1 else [axes]
    for i, (img, grayscale_cam) in enumerate(zip(processed_imgs, grayscale_cams)):
        # 调整 alpha 值，增加热力图的显著性
        cam_img = show_cam_on_image(img, grayscale_cam, use_rgb=True)

        # 在子图中显示 Grad-CAM 结果
        axes[i].imshow(cam_img)
        axes[i].set_title(f'{classes[labels[i]]}')
        axes[i].axis('off')  # 隐藏坐标轴

    plt.tight_layout()
    plt.show()
