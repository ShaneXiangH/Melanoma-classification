from torchvision import models
from torch import nn, optim
import torch
import os


class Mobilenetv2(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        os.environ['TORCH_HOME'] = '.{0}pretrained_models{0}mobilenet_pth'.format(os.sep)
        self.net = models.mobilenet_v2(True)
        model = nn.Linear(1280, num_classes, True)
        nn.init.normal_(model.weight, 0, 0.01)
        nn.init.zeros_(model.bias)
        self.net.classifier = nn.Sequential(
            nn.Dropout(0.2),
            model,
            nn.Softmax(1)
        )

    def forward(self, img):
        out = self.net(img)
        return out


def test():
    net = Mobilenetv2(2)
    x = torch.randn(5, 3, 224, 224)
    target = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lr=0.01, params=net.parameters())
    for epoch in range(100):
        optimizer.zero_grad()

        y = net(x)
        loss = criterion(y, target)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                cor = total = 0
                x_test = torch.randn(5, 3, 224, 224)
                target_test = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]])
                y_test = net(x_test)
                _, pred = torch.max(y_test.data, 1)
                total += target_test.size(0)
                cor += (pred == target_test).sum().item()

                print("test_acc:", round(cor / total, 6))


test()
