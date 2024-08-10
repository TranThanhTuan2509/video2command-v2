import torch.nn as nn

class Classification(nn.Module):
    def __init__(self, num_classes=41):
        super().__init__()
        self.conv1 = self._make_block(30, 2048)   # Output's shape: B, 16, 112, 112
        self.conv2 = self._make_block(2048, 1024)  # Output's shape: B, 32, 56, 56
        self.conv3 = self._make_block(1024, 512)  # Output's shape: B, 64, 28, 28

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512*256, out_features=256),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def _make_block(self, in_channels, out_channels, kernel_size=1):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
