import os
import random

def remove_random_images(directory, num_to_remove):
    """
    Rimuove randomicamente un numero specificato di immagini da una directory.

    Parameters:
        directory (str): Il percorso della directory contenente le immagini.
        num_to_remove (int): Il numero di immagini da rimuovere.
    """
    try:
        # Ottieni una lista di tutti i file nella directory
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

        # Assicurati che ci siano abbastanza file per la rimozione
        if num_to_remove > len(files):
            print(f"Errore: ci sono solo {len(files)} file nella directory, ma hai chiesto di rimuoverne {num_to_remove}.")
            return

        # Seleziona randomicamente i file da rimuovere
        files_to_remove = random.sample(files, num_to_remove)

        # Rimuovi i file selezionati
        for file in files_to_remove:
            file_path = os.path.join(directory, file)
            os.remove(file_path)
            print(f"Rimosso: {file_path}")

        print(f"Rimozione completata: {num_to_remove} file rimossi da {directory}.")

    except Exception as e:
        print(f"Si è verificato un errore: {e}")

# Esempio di utilizzo:
# Specifica il percorso della directory e il numero di immagini da rimuovere
directory_path = "FER2013/train/fear"
number_of_images_to_remove = 97

remove_random_images(directory_path, number_of_images_to_remove)
