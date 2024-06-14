
import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FCN, self).__init__()

        self.conv1 = nn.Conv1d(input_size, 128, kernel_size=8, stride= 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, stride= 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, stride= 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x = x.unsqueeze(1)
        # x = x.permute(0, 2, 1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))

        x = torch.mean(x, dim=2)  # Global average pooling

        x = self.fc(x)

        return x
