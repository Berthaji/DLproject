import torch
import pandas as pd
import os
from evaluation import validate_model

def freeze_resnet_layers(model):
   
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the penultimate
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Unfreeze the last
    for param in model.fc.parameters():
        param.requires_grad = True


def freeze_efficientnet_layers(model):
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last few layers
    for param in model.features[-1].parameters():
        param.requires_grad = True
    
    for param in model.classifier.parameters():
        param.requires_grad = True


def freeze_mobilenet_layers(model):
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last few layers
    for param in model.features[-3:].parameters(): # sblocca di più rispetto agli altri
        param.requires_grad = True
    
    for param in model.classifier.parameters(): 
        param.requires_grad = True


def train_model_TL(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, output_path='results/training_results_resnet.csv', device="cpu", validate=True, model_name="resnet"):
   
    if model_name == "resnet":
        freeze_resnet_layers(model)
    if model_name == "efficientnet":
        freeze_efficientnet_layers(model)
    if model_name == "mobilenet":
        freeze_mobilenet_layers(model)


    # Creare la cartella di output se non esiste
    output_folder = os.path.dirname(output_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # DataFrame per salvare i risultati
    results = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Train Accuracy', 'Validation Accuracy'])

    for epoch in range(num_epochs):
        model.train()  # Modalità training
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # Azzerare i gradienti
            
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Calcolare la perdita
            loss.backward()  # Backward pass
            optimizer.step()  # Aggiornare i pesi

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calcolare le metriche di training
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        if validate:
            val_accuracy = validate_model(model, val_loader, device)

            # Log dei risultati
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, "
                f"Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")

            # Salvare i risultati nell'oggetto DataFrame
            results = pd.concat([
                results,
                pd.DataFrame({
                    'Epoch': [epoch+1], 
                    'Train Loss': [train_loss], 
                    'Train Accuracy': [train_accuracy], 
                    'Validation Accuracy': [val_accuracy]
                })
            ], ignore_index=True)

            # Salvare i risultati nel file CSV
            results.to_csv(output_path, index=False)
    print(f"Risultati validazione salvati in: {output_path}")