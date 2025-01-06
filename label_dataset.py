import os
import pandas as pd

# Percorsi delle directory
happy_dir_train = "FER2013/train/happy"
sad_dir_train = "FER2013/train/sad"

# Creazione di liste per memorizzare i dati
data = []

# Itera sui file nella directory "happy" e assegna l'etichetta 1
for file_name in os.listdir(happy_dir_train):
    data.append([file_name, 1])

# Itera sui file nella directory "sad" e assegna l'etichetta 0
for file_name in os.listdir(sad_dir_train):
    data.append([file_name, 0])

# Creazione di un DataFrame
df = pd.DataFrame(data, columns=["Image_Name", "Label"])

# Salvataggio in un file CSV
output_path = "image_labels_train.csv"
df.to_csv(output_path, index=False)

print(f"File CSV creato con successo: {output_path}")