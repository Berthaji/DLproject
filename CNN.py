import torch.nn.functional as F
import torch.nn as nn
import torch


# modifiche ai layer della CNN, al dropout e ai pesi della cross-entropy loss nella funzione di test per le performance
# qui dovrebbe già andare bene

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # Primo layer convolutivo
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Secondo layer convolutivo
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Terzo layer convolutivo
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Max pooling per ridurre la dimensione
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout per evitare overfitting
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layer
        self.fc1 = nn.Linear(128 * 6 * 6, 512)  # Per immagini 48x48
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Primo livello convolutivo + batchnorm + ReLU
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Secondo livello convolutivo + batchnorm + ReLU
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Terzo livello convolutivo + batchnorm + ReLU
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected + ReLU
        x = F.relu(self.fc1(x))

        # Dropout
        x = self.dropout(x)

        # Output layer
        x = self.fc2(x)
        return x

# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=2):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)  # Batch normalization per il primo layer convolutivo
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)  # Batch normalization per il secondo layer convolutivo
#         self.dropout = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(64 * 12 * 12, 512)
#         self.fc2 = nn.Linear(512, num_classes)

#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))  # BatchNorm + ReLU
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))  # BatchNorm + ReLU
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x