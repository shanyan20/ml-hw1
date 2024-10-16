import netron

if __name__ == '__main__':
    modelData = "./model/resnet_18epochs.pth"  # 定义模型数据保存的路径
    netron.start(modelData)  # 输出网络结构
