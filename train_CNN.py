import torch
import pandas as pd
import os
from evaluation import validate_model

def train_model_CNN(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, output_path='results/training_results.csv', device="cpu", validate=True):
    output_folder = 'results'

    # Crea la cartella 'results' se non esiste
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Crea un DataFrame vuoto per salvare i risultati
    results = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Train Accuracy', 'Validation Accuracy'])

    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # Reset gradients
            outputs = model(images)  # Pass through the model
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Calculate gradients
            optimizer.step()  # Update weights

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        if validate:
            val_accuracy = validate_model(model, val_loader, device)

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, "
                f"Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")
            
            # Aggiungi i risultati al DataFrame
            results = pd.concat([
                results, 
                pd.DataFrame({'Epoch': [epoch+1], 'Train Loss': [train_loss], 
                            'Train Accuracy': [train_accuracy], 'Validation Accuracy': [val_accuracy]})
            ], ignore_index=True)
        
            results.to_csv(output_path, index=False)
            print(f"Risultati validazione salvati in: {output_path}")
    print("Addestramento completato.")