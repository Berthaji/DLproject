import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from CNN import SimpleCNN
from train_CNN import train_model_CNN
from eval_CNN import evaluate_model

# Definizione delle trasformazioni per le immagini
transform = transforms.Compose([
    transforms.Resize((48, 48)),  # Ridimensionamento delle immagini a 48x48
    transforms.ToTensor(),        # Conversione in tensore
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalizzazione
])

# Caricamento dei dati di addestramento
train_dir = "FER2013/train"  # La directory che contiene le sottocartelle "happy" e "sad"
# imageFolder assegna le etichette in automatico alle immagini in base alla cartella dove si trovano
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)

# Suddivisione dei dati in set di addestramento e di validazione (80% training, 20% validation)
train_data, val_data = train_test_split(train_dataset.samples, test_size=0.2, random_state=42)

# Creazione dei DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Creazione del modello
model = SimpleCNN()

# Funzione di perdita (cross-entropy per classificazione binaria)
criterion = nn.CrossEntropyLoss()

# Ottimizzatore (Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Addestramento del modello
train_model_CNN(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

# Salvataggio del modello
torch.save(model.state_dict(), "happy_sad_model.pth")

# Caricamento e valutazione del modello sui dati di test
class TestDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.df = pd.read_csv(csv_file)  # Carica il CSV con le etichette
        self.image_names = self.df['Image_Name'].values
        self.labels = self.df['Label'].values

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Percorso della cartella di test
test_dir = "FER2013/test"

# Carica il dataset di test utilizzando ImageFolder
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# Creazione del DataLoader per i dati di test
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Carica il modello salvato e valuta sui dati di test
model.load_state_dict(torch.load("happy_sad_model.pth"))
evaluate_model(model, test_loader)