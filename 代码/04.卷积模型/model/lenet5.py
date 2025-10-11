import torch
from torch import nn
from torch.nn import functional as F


class Lenet5(nn.Module):
    """
    for cifar10 dataset.
    """

    def __init__(self):
        super(Lenet5, self).__init__()
        self.conv_unit = nn.Sequential(
            # x: [b, 3, 32, 32] => [b, 16, ]
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        # fc unit
        self.fc_unit = nn.Sequential(
            nn.Linear(32 * 5 * 5, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        """
        :param x: [b, 3, 32, 32]
        :return:
        """
        batchsz = x.size(0)
        # [b, 3, 32, 32] => [b, 16, 5, 5]
        x = self.conv_unit(x)
        # [b, 16, 5, 5] => [b, 16*5*5]
        x = x.view(batchsz, 32 * 5 * 5)
        # [b, 16*5*5] => [b, 10]
        logits = self.fc_unit(x)
        return logits


def main():
    net = Lenet5()
    tmp = torch.randn(2, 3, 32, 32)
    out = net(tmp)
    print('lenet out:', out.shape)


if __name__ == '__main__':
    main()
