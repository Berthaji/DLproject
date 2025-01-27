import os
from PIL import Image
import random
from PIL import ImageFilter
import torch
from torchvision import transforms

# classe per applicazione di un filtro mediano (denoising)
class RandomDenoising:
    def __init__(self, probability=0.5):
        self.probability = probability
    
    def __call__(self, image):
        if random.random() < self.probability:
            # Applica un filtro mediano per denoising
            image = image.filter(ImageFilter.MedianFilter(size=3))
        return image

# Configurazione
source_directory = "FER2013_multiclasse/train/angry"  # Percorso della cartella con le immagini originali
target_number = 3220
image_size = (48, 48)  # Dimensione delle immagini (puoi modificarla se necessario)

# Definizione delle trasformazioni per la data augmentation
transform = transforms.Compose([
    transforms.RandomRotation(30),  # Ruotare le immagini casualmente fino a 30 gradi
    transforms.RandomHorizontalFlip(),  # Flip orizzontale casuale
    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),  # Crop casuale e ridimensionamento
    RandomDenoising(probability=0.6), #denoising con probabilità del 60%
    transforms.ToTensor(),  # Converti in tensore
])

# Lista delle immagini originali
image_files = os.listdir(source_directory)
if len(image_files) == 0:
    print("Nessuna immagine trovata nella directory originale.")
    exit()

# Controllo del numero di immagini originali
num_original_images = len(image_files)
print(f"Trovate {num_original_images} immagini originali.")

# Calcolo del numero di augmentazioni per immagine
augmentations_per_image = target_number // num_original_images + 1

# Inizializzazione del contatore delle immagini salvate
total_saved_images = 0

# Creazione delle immagini augmentate
for file_name in image_files:
    if total_saved_images >= target_number:
        break  # Interrompi quando raggiungi il numero desiderato

    # Caricamento dell'immagine originale
    img_path = os.path.join(source_directory, file_name)
    img = Image.open(img_path)
    img = img.convert('RGB')  # Assicurati che l'immagine sia in formato RGB

    # Applica la trasformazione per ottenere un'immagine augmentata
    for i in range(augmentations_per_image):
        augmented_img = transform(img)  # Applicazione della trasformazione

        # Salva l'immagine augmentata nella cartella originale
        augmented_file_name = f"augmented_{total_saved_images + 1}_{file_name}"
        augmented_img_path = os.path.join(source_directory, augmented_file_name)
        
        # Converti il tensore in un'immagine PIL e salva
        augmented_img_pil = transforms.ToPILImage()(augmented_img)
        augmented_img_pil.save(augmented_img_path)
        
        total_saved_images += 1

        if total_saved_images >= target_number:
            break  # Interrompi se hai creato abbastanza immagini

print(f"Creato un totale di {total_saved_images} immagini augmentate nella directory: {source_directory}.")

