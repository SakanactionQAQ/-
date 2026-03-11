import torch
import torch.nn as nn

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1,
                          stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):

        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)

        out = torch.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_c, out_c, blocks, stride):

        layers = []
        layers.append(ResidualBlock(in_c, out_c, stride))

        for i in range(1, blocks):
            layers.append(ResidualBlock(out_c, out_c))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = torch.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


import os
import cv2
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, root):

        self.images = []
        self.labels = []

        classes = os.listdir(root)

        for label, c in enumerate(classes):

            path = os.path.join(root, c)

            for img in os.listdir(path):

                self.images.append(os.path.join(path, img))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.images[idx]

        img = cv2.imread(img_path)
        img = cv2.resize(img, (224,224))

        img = img / 255.0
        img = torch.tensor(img).permute(2,0,1).float()

        label = torch.tensor(self.labels[idx])

        return img, label


model = ResNet()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()

for epoch in range(20):

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

writer.add_scalar("loss", loss.item(), step)

import time

start = time.time()

# training

end = time.time()

print("training time:", end-start)