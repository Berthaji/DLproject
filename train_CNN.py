import torch

def train_model_CNN(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # Imposta il modello in modalità training
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()  # Azzerare i gradienti
            outputs = model(images)  # Passa l'immagine attraverso il modello
            loss = criterion(outputs, labels)  # Calcola la perdita
            loss.backward()  # Calcola il gradiente
            optimizer.step()  # Aggiorna i pesi

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Accuracy: {100 * correct/total}%")

        # Valutazione sul set di validazione
        model.eval()  # Imposta il modello in modalità eval
        with torch.no_grad():
            val_correct = 0
            val_total = 0
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

            print(f"Validation Accuracy: {100 * val_correct / val_total}%")