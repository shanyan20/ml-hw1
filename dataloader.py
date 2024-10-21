import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# 定义DataLoader函数
def get_dataloader(batch_size, image_size=32):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='F:\study\master1\mechine_learning\hw1\data',
        train=True,
        download=False,
        transform=transform,
    )

    testset = torchvision.datasets.CIFAR10(
        root='F:\study\master1\mechine_learning\hw1\data',
        train=False,
        download=False,
        transform=transform,
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    return trainloader, testloader


def get_augmented_dataloader(batch_size, image_size=32):
    transform = transforms.Compose([
        # transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),  # 随机裁剪并调整大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # 颜色抖动
        transforms.RandomRotation(15),  # 随机旋转
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
        # transforms.RandomErasing(p=0.5),  # 随机擦除，50%概率
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=False,
        transform=transform,
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=False,
        transform=transform,
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    return trainloader, testloader


def imshow(img):
    # 反归一化，将数据重新映射到0-1之间
    img = img / 2 + 0.5
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    # trainloader, testloader = get_dataloader(4, 224)
    trainloader, testloader = get_augmented_dataloader(4, 224)
    # CIFAR-10的类别标签
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # 从数据集中获取一个 batch 的样本
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # 显示图像
    imshow(torchvision.utils.make_grid(images))

    # 打印对应的标签
    print(' '.join(f'{classes[labels[j]]}' for j in range(4)))
