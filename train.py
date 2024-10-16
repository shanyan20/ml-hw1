import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import numpy as np


# 定义训练函数
# 定义训练函数，支持中断继续
def train(model, device, trainloader, optimizer, criterion, start_epoch, num_epochs, model_path, loss_path, losses):
    epoch = 0
    try:
        for epoch in range(start_epoch, num_epochs):
            running_loss = 0.0
            model.train()  # 训练模式
            for i, (inputs, labels) in enumerate(trainloader, 0):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # 输出训练进度
                if (i + 1) % (len(trainloader) // 10) == 0:
                    print(
                        f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {loss.item():.4f}')

            epoch_loss = running_loss / len(trainloader)
            losses.append(epoch_loss)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.4f}')

            # 保存模型
            torch.save(model.state_dict(), model_path)

        # 绘制损失曲线
        plt.plot(range(1, num_epochs + 1), losses, marker='o')
        plt.title('Training Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid()
        plt.savefig(loss_path)
        plt.show()
        # 删除检查点文件
        if os.path.exists('./checkpoint.pth'):
            os.remove('./checkpoint.pth')
            print(f'Checkpoint file ./checkpoint.pth deleted.')
    except KeyboardInterrupt:
        print('Training interrupted. Saving checkpoint...')
        save_checkpoint(model, optimizer, epoch, './checkpoint.pth', losses)


# 定义测试函数，输出精确率和混淆矩阵图片
def test(model, device, testloader, classes, cm_path):
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    model.eval()  # 设置为评估模式
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 记录所有标签和预测结果
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # 计算精确率
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 使用matplotlib绘制混淆矩阵
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # 设置标签
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # 设置轴标签和标题
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # 在每个方格内添加数值
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')

    # 保存混淆矩阵图片
    plt.savefig(cm_path)
    print(f'Confusion matrix saved to {cm_path}')
    plt.show()


# 定义保存检查点的函数
def save_checkpoint(model, optimizer, epoch, path, losses):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses
    }
    torch.save(checkpoint, path)
    print(f'Checkpoint saved at epoch {epoch + 1}')


# 定义加载检查点的函数
def load_checkpoint(path, model, optimizer):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        losses = checkpoint.get('losses', [])  # 如果没有保存损失列表，则返回空列表
        print(f'Checkpoint loaded. Resuming from epoch {start_epoch}')
        return start_epoch, losses
    else:
        print('No checkpoint found. Starting from scratch.')
        return 0, []

