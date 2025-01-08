import torch
import torch.nn as nn

# Creazione del modello CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # Convoluzione: 3 canali (RGB), 32 filtri
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling con dimensione 2x2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Convoluzione: 32 canali, 64 filtri
        self.fc1 = nn.Linear(64 * 12 * 12, 512)  # Fully connected layer 1
        self.fc2 = nn.Linear(512, 2)  # Fully connected layer 2 (2 classi: happy, sad)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Applicazione convoluzione + ReLU + max pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Applicazione convoluzione + ReLU + max pooling
        x = x.view(-1, 64 * 12 * 12)  # Flatten per il layer fully connected (12x12 dopo il pooling)
        x = torch.relu(self.fc1(x))  # Passaggio attraverso il primo layer fully connected
        x = self.fc2(x)  # Passaggio attraverso l'ultimo layer per ottenere le probabilità
        return x
