import torch.nn as nn

class Cnn_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Linear()

    def foward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x

        