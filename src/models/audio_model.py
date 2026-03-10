import torch
import torch.nn as nn


class AudioModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(1,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )

        self.lstm = nn.LSTM(

            input_size=64,
            hidden_size=128,
            batch_first=True,
            bidirectional=True

        )

        self.fc = nn.Linear(256,256)

    def forward(self,x):

        x = self.cnn(x)

        x = x.mean(dim=2)

        x = x.permute(0,2,1)

        x,_ = self.lstm(x)

        x = x[:,-1,:]

        x = torch.relu(self.fc(x))

        return x