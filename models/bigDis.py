import torch.nn as nn


class BigDis(nn.Module):
    def __init__(self):
        super(BigDis, self).__init__()

        self.model = nn.Sequential(
            # Increase channel depth and use spectral normalization
            nn.utils.spectral_norm(nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2),

            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # Add more layers with increasing channel depth
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # Optional: Add Self-Attention Layer here
            # SelfAttention(512),

            nn.utils.spectral_norm(nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class MidDis(nn.Module):
    def __init__(self):
        super(MidDis, self).__init__()

        self.model = nn.Sequential(
            # Increase channel depth and use spectral normalization
            nn.utils.spectral_norm(nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2),

            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # Add more layers with increasing channel depth
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # Optional: Add Self-Attention Layer here
            # SelfAttention(512),

            # nn.utils.spectral_norm(nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)),
            # nn.BatchNorm2d(1024),
            # nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class JuniorDis(nn.Module):
    def __init__(self):
        super(JuniorDis, self).__init__()

        self.model = nn.Sequential(
            # Increase channel depth and use spectral normalization
            nn.utils.spectral_norm(nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2),

            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # Add more layers with increasing channel depth
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)