'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models as models

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from process_data import get_data_transforms, SIIM_ISIC

from sklearn.metrics import roc_curve, auc

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.025, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train, transform_test = get_data_transforms()

trainset = SIIM_ISIC(transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)


testset = SIIM_ISIC(train=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)

classes = ('true', 'false')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()+
# net = RegNetX_200MF()

# net = EfficientNetB0()
net = net.to(device)
cudnn.benchmark = True
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('./checkpoint/ckpt.pth')
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

parameters = filter(lambda p: p.requires_grad, net.parameters())
optimizer = optim.SGD(parameters, lr=args.lr,
                      momentum=0.9, weight_decay=3e-4)

total_epochs = 200
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(total_epochs))


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, meta,  targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device=device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            scores = nn.functional.softmax(outputs, dim=1)
            positive_scores = scores[:, 1]
            # print(positive_scores)
            # print(targets)
            # auc = roc_auc_score(targets.to('cpu'), positive_scores.to('cpu'))
            fpr, tpr, thresholds = roc_curve(targets.to('cpu'), positive_scores.to('cpu'), pos_label = 1)
            auc_score = auc(fpr, tpr)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Auc: %.3f'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, auc_score))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, meta,  targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            scores = nn.functional.softmax(outputs, dim=1)

            positive_scores = scores[:, 1]
            # auc = roc_auc_score(targets.to('cpu'), positive_scores.to('cpu'))
            fpr, tpr, thresholds = roc_curve(targets.to('cpu'), positive_scores.to('cpu'), pos_label = 1)
            auc_score = auc(fpr, tpr)

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Auc: %.3f'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total, auc_score))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    print("best acc: ", best_acc)


for epoch in range(start_epoch, start_epoch+total_epochs):
    scheduler.step()
    print('epoch ', epoch, 'lr ', scheduler.get_lr()[0])
    train(epoch)
    test(epoch)
