import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, models
from torchvision.models import ResNet18_Weights
from torchvision.transforms import Compose
from CNN import SimpleCNN
from train_CNN import train_model_CNN
from train_TL import train_model_TL
from utils import check_images, FocalLoss
from evaluation import evaluate_model


def train_and_test_CNN(
        train_dir, 
        test_dir, 
        num_epochs=10, 
        batch_size=32, 
        model_save_path="model.pth",
        validation_results_path="results/validation_results.csv",
        test_results_path="results/test_results.csv",
        mode="binary", 
        loss_function="crossentropy",
        device="cpu",
        validate="True"
        ):

    # Definizione delle trasformazioni per le immagini
    transform = Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((48, 48)),  # Ridimensionamento delle immagini a 48x48
        transforms.ToTensor(),        # Conversione in tensore
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizzazione
    ])
    
    # Controlla le immagini nella cartella di training
    check_images(train_dir)

    # Caricamento del dataset di training con ImageFolder
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    train_loader_no_validation = train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    # Suddivisione del dataset in training e validation (80% training, 20% validation)
    dataset_size = len(train_dataset)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    split = int(np.floor(0.2 * dataset_size))  # Calcolo 20% per la validazione
    train_indices, val_indices = indices[split:], indices[:split]

    # Creazione dei subset
    train_data = Subset(train_dataset, train_indices)
    val_data = Subset(train_dataset, val_indices)

    # Creazione dei DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Debug: verifica la struttura dei batch
    for images, labels in train_loader:
        print(f"Batch immagini: {images.shape}, Batch etichette: {labels.shape}")
        break  # Verifica solo il primo batch

    if mode=="binary":
        model = SimpleCNN(num_classes=2)
        weights = torch.tensor([1.0, 1.5], dtype=torch.float32).to(device)  # Peso maggiore per la classe 1 per stabilizzare prestazioni
    if mode=="multiclass":
        model = SimpleCNN(num_classes=4)
        weights = torch.tensor([2.0, 2.5, 1.5, 3.0], dtype=torch.float32).to(device)

    # Sposta il modello sul dispositivo (MPS, CUDA, CPU)
    model = model.to(device).float()

    # Funzione di perdita 
    if loss_function=="crossentropy":
        criterion = nn.CrossEntropyLoss(weight=weights).to(device)
    if loss_function=="focal":  # utile quando ci sono classi difficili
        criterion = FocalLoss(alpha=1, gamma=1.5, weight=weights, reduction='mean').to(device)

    # Ottimizzatore (Adam)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if validate:
        train_model_CNN(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, output_path=validation_results_path, device=device, validate=True)
    else:
        train_model_CNN(model, train_loader_no_validation, val_loader, criterion, optimizer, num_epochs=num_epochs, output_path=validation_results_path, device=device, validate=False)


    # Salvataggio del modello addestrato
    torch.save(model.state_dict(), model_save_path)
    print(f"Modello salvato in: {model_save_path}")

    # Controlla le immagini nella cartella di test
    check_images(test_dir)

    # Carica il dataset di test con ImageFolder
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # Creazione del DataLoader per i dati di test
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Carica il modello salvato e valuta sui dati di test
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))

    if mode=="binary":
        evaluate_model(model, test_loader, 2, test_results_path, device)
    if mode=="multiclass":
        evaluate_model(model, test_loader, 4, test_results_path, device)

    print("Test completato con successo.")




# Funzione di addestramento e test per modelli transfer learning
def train_and_test_TL(
        train_dir,
        test_dir,
        num_epochs=5,
        batch_size=32,
        model_save_path="resnet_model.pth",
        validation_results_path="results/validation_results_resnet.csv",
        test_results_path="results/test_results_resnet.csv",
        mode="binary",
        loss_function="crossentropy",
        device="cpu",
        validate = True,
        model_name= "resnet"
    ):
    
    # Definizione delle trasformazioni per le immagini
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Dimensione di input per ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizzazione standard per ResNet
    ])

    # Caricamento del dataset di training
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    train_loader_no_validation = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    # Suddivisione del dataset in training e validation (80% training, 20% validation)
    dataset_size = len(train_dataset)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    split = int(np.floor(0.2 * dataset_size))  # Calcolo 20% per la validazione
    train_indices, val_indices = indices[split:], indices[:split]

    train_data = Subset(train_dataset, train_indices)
    val_data = Subset(train_dataset, val_indices)

    # Creazione dei DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    for images, labels in train_loader:
        print(f"Batch immagini: {images.shape}, Batch etichette: {labels.shape}")
        break 


    if model_name == "resnet":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = model.fc.in_features
    elif model_name == "efficientnet":
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        num_features = model.classifier[1].in_features
    elif model_name == "mobilenet":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        num_features = model.classifier[1].in_features  # Cambiato per MobileNetV2

    # CAPIRE SE QUESTI PESI FUNZIONANO PER TUTTI E TRE I MODELLI
    if mode == "binary":
        if model_name == "resnet":
            model.fc = nn.Linear(num_features, 2)  # ResNet ha 'fc' come layer finale
        elif model_name == "efficientnet":
            model.classifier[1] = nn.Linear(num_features, 2)  # EfficientNet ha 'classifier' come layer finale
        elif model_name == "mobilenet":
            model.classifier[1] = nn.Linear(num_features, 2)  # MobileNet ha 'classifier' come layer finale
        weights = torch.tensor([1.0, 1.5], dtype=torch.float32)
    elif mode == "multiclass":
        if model_name == "resnet":
            model.fc = nn.Linear(num_features, 4)  # ResNet ha 'fc' come layer finale
        elif model_name == "efficientnet":
            model.classifier[1] = nn.Linear(num_features, 4)  # EfficientNet ha 'classifier' come layer finale
        elif model_name == "mobilenet":
            model.classifier[1] = nn.Linear(num_features, 4)  # MobileNet ha 'classifier' come layer finale
        weights = torch.tensor([2.0, 1.5, 1.0, 1.3], dtype=torch.float32)

    # Sposta il modello e i pesi sul dispositivo
    model = model.to(device).float()  # Forza float32
    weights = weights.to(device).float()  # Aggiunto: sposta i pesi sul dispositivo

    # Configura la funzione di perdita
    if loss_function == "crossentropy":
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif loss_function == "focal":
        criterion = FocalLoss(alpha=1, gamma=1.5, weight=weights, reduction='mean')

    # Ottimizzatore
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)

    if validate:
        train_model_TL(
            model, 
            train_loader, 
            val_loader, 
            criterion, 
            optimizer, 
            num_epochs=num_epochs, 
            output_path=validation_results_path, 
            device=device,
            validate=True,
            model_name=model_name
        )
    else:
          train_model_TL(
            model, 
            train_loader_no_validation, 
            val_loader, 
            criterion, 
            optimizer, 
            num_epochs=num_epochs, 
            output_path=validation_results_path, 
            device=device,
            validate=False,
            model_name=model_name
        )

    # Salvataggio del modello addestrato
    torch.save(model.state_dict(), model_save_path)
    print(f"Modello salvato in: {model_save_path}")

    # Caricamento del dataset di test
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Carica il modello salvato per il test
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    model = model.to(device).float()  # Assicurati che il modello sia di tipo float32

    # Valutazione sul test set
    if mode == "binary":
        evaluate_model(model, test_loader, 2, test_results_path, device)
    elif mode == "multiclass":
        evaluate_model(model, test_loader, 4, test_results_path, device)

    print("Test completato con successo.")



def train_and_test_all_TL_models(
        train_dir,
        test_dir,
        num_epochs=5,
        batch_size=32,
        model_save_path="resnet_model.pth",
        validation_results_path="results/validation_results_resnet.csv",
        test_results_path="results/test_results_resnet.csv",
        mode="binary",
        loss_function="crossentropy",
        device="cpu",
        validate=True
):
    model_names = ["resnet", "efficientnet", "mobilenet"]
    
    for name in model_names:
        train_and_test_TL(
            train_dir,
            test_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            model_save_path=model_save_path.replace("resnet", name),
            validation_results_path=validation_results_path.replace("resnet", name),
            test_results_path=test_results_path.replace("resnet", name),
            mode=mode,
            loss_function=loss_function,
            device=device,
            validate=validate,
            model_name=name
        )
