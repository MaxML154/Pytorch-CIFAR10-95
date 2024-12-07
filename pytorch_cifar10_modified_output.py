import torchvision.transforms as transforms
import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import argparse

from backbones.ResNet import ResNet18

# 指定GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 用于计算GPU运行时间
def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

# Training
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    train_acc = 0
    # 开始迭代每个batch中的数据
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # 计算损失
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 计算准确率
        train_acc = correct / total

        # 每训练100个batch打印一次训练集的loss和准确率
        if (batch_idx + 1) % 100 == 0:
            print(f'[INFO] Epoch-{epoch+1}-Batch-{batch_idx+1}: Train Loss-{loss.item():.4f}, Accuracy-{train_acc:.4f}')

    total_train_acc.append(train_acc)

# Testing
def test(epoch, ckpt):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_acc = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        test_acc = correct / total
        print(f'[INFO] Epoch-{epoch+1}: Test Accuracy: {test_acc:.4f}')

    total_test_acc.append(test_acc)

    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, ckpt)
        best_acc = acc

# 预测并显示图像
def predict_images():
    model.eval()
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    _, predicted = outputs.max(1)

    # 显示5张图像和其预测类别
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        ax = axes[i]
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        ax.imshow(img)
        ax.set_title(f'Pred: {classes[predicted[i]]} ({classes[labels[i]]})')
        ax.axis('off')
    plt.savefig('output/ResNet18-CIFAR10-Prediction-Result.jpg')
    plt.show()

if __name__ == '__main__':
    # 设置超参
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--T_max', type=int, default=100)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/ResNet18-CIFAR10.pth')
    opt = parser.parse_args()

    # 设置相关参数
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 设置数据增强
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 加载CIFAR10数据集
    trainset = torchvision.datasets.CIFAR10(
        root=opt.data, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=opt.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root=opt.data, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    # 加载模型
    print('==> Building model..')
    model = ResNet18().to(device)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # 加载之前训练的参数
    if opt.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(opt.checkpoint)
        model.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # 设置损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=opt.lr,
                          momentum=0.9, weight_decay=5e-4)

    # 余弦退火有序调整学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.T_max)

    # 记录training和testing的acc
    total_test_acc = []
    total_train_acc = []

    # 记录训练时间
    tic = time_sync()

    # 开始训练
    for epoch in range(opt.epochs):
        train(epoch)
        test(epoch, opt.checkpoint)
        scheduler.step()

    # 绘制损失和准确率曲线
    plt.figure()
    plt.plot(range(opt.epochs), total_train_acc, label='Train Accuracy')
    plt.plot(range(opt.epochs), total_test_acc, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('ResNet18-CIFAR10 Accuracy')
    plt.legend()
    plt.savefig('output/ResNet18-CIFAR10-Accuracy.jpg')
    plt.show()

    # 输出best_acc
    print(f'Best Acc: {best_acc * 100}%')
    toc = time_sync()
    t = (toc - tic) / 3600
    print(f'Training Done. ({t:.3f}s)')

    # 预测图像
    predict_images()
