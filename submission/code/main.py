'''
Author: All group members
'''
import torch.optim as optim
from process_data import get_data_transforms, SIIM_ISIC
import torch.backends.cudnn as cudnn
import os
import argparse
from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='CS324 Final')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
args = parser.parse_args()

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

print('Data preprocessing...')
transform_train, transform_test = get_data_transforms(size=224)

trainset = SIIM_ISIC(transform=transform_train)
trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=4,
        num_workers=16,
        shuffle=True,
        pin_memory=True
    )

testset = SIIM_ISIC(type='validate', transform=transform_test)
testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=4,
        num_workers=16,
        shuffle=False,
        pin_memory=True
    )

print('Model initializing...')
net = EfficientNetB0()
net = net.to(device)
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, 0.00001)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, meta, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device=device, dtype=torch.int64)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, meta, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device=device, dtype=torch.int64)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

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


for epoch in range(start_epoch, start_epoch+500):
    train(epoch)
    test(epoch)
