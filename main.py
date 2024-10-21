import os
from model import *
from dataloader import *
from train import *
import torch.optim as optim

# 检查CUDA可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
batch_size = 64
num_epochs = 60
learning_rate = 0.002

# 模型和存档点保存路径
checkpoint_path = './checkpoint.pth'

# 模型保存路径
cnn_model_path = 'cnn'
resnet_model_path = 'resnet'

# 主程序入口
if __name__ == '__main__':
    # 选择使用的模型：CNN或ResNet50
    use_cnn = True  # 设置为True时使用CNN模型，False时使用ResNet50
    only_test = False  # 是否仅在测试集上测试模型性能
    # 获取数据加载器
    image_size = 224 if not use_cnn else 32
    trainloader, testloader = get_dataloader(batch_size, image_size)
    # trainloader, testloader = get_augmented_dataloader(batch_size, image_size)
    # CIFAR-10的类别标签
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    if use_cnn:
        model = CNN().to(device)
        model_path = './' + cnn_model_path + '.pth'
        # model_path = './model/cnn_augmented.pth'
        loss_path = './picture/' + cnn_model_path + '_loss.png'
        cm_path = './picture/' + cnn_model_path + '_confusion_matrix.png'

    else:
        model = get_resnet50().to(device)
        model_path = './' + resnet_model_path + '.pth'
        # model_path = './model/resnet_18epochs.pth'
        loss_path = './picture/' + resnet_model_path + '_loss.png'
        cm_path = './picture/' + resnet_model_path + '_confusion_matrix.png'

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # 加载检查点（如果存在）
    start_epoch, losses = load_checkpoint(checkpoint_path, model, optimizer)

    # 检查是否存在已保存的模型
    if only_test:
        print(f'Loading model from {model_path}')
        model.load_state_dict(torch.load(model_path))
    else:
        # 训练模型
        train(model, device, trainloader, optimizer, criterion, start_epoch, num_epochs, model_path, loss_path, losses)

    # 测试模型并保存混淆矩阵为PNG
    test(model, device, testloader, classes, cm_path)


