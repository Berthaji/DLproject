import os
import pandas as pd

# Percorsi delle directory
happy_dir = "FER2013/test/happy"
sad_dir = "FER2013/test/sad"

# Creazione di liste per memorizzare i dati
data = []
happy_data = []
sad_data = []

# Itera sui file nella directory "happy" 
for file_name in os.listdir(happy_dir):
    data.append([file_name, 0])
    happy_data.append([file_name, 0])  # Aggiungi solo i file happy

# Itera sui file nella directory "sad" 
for file_name in os.listdir(sad_dir):
    data.append([file_name, 1])
    sad_data.append([file_name, 1])  # Aggiungi solo i file sad

# Creazione di DataFrame
df_all = pd.DataFrame(data, columns=["Image_Name", "Label"])
df_happy = pd.DataFrame(happy_data, columns=["Image_Name", "Label"])
df_sad = pd.DataFrame(sad_data, columns=["Image_Name", "Label"])

# Salvataggio dei file CSV
output_all_path = "all_image_labels.csv"
output_happy_path = "happy_labels.csv"
output_sad_path = "sad_labels.csv"

df_all.to_csv(output_all_path, index=False)
df_happy.to_csv(output_happy_path, index=False)
df_sad.to_csv(output_sad_path, index=False)

print(f"File CSV creati con successo:")
print(f"  - Happy e sad: {output_all_path}")
print(f"  - Solo happy: {output_happy_path}")
print(f"  - Solo sad: {output_sad_path}")