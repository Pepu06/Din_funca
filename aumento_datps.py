import nlpaug.augmenter.word as naw
import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv('mati.csv')

# Inicializar el aumentador de palabras usando parafraseo
aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")

# Función para parafrasear el texto
def paraphrase_text(text, augmenter, num_variations=3):
    augmented_texts = [augmenter.augment(text) for _ in range(num_variations)]
    return augmented_texts

# Generar más datos para cada entrada en el CSV
new_data = []

for index, row in df.iterrows():
    text = row['text']
    paraphrases = paraphrase_text(text, aug)
    for para in paraphrases:
        new_data.append(para)

# Crear un nuevo DataFrame con los datos aumentados
new_df = pd.DataFrame(new_data, columns=['text'])

# Guardar el nuevo DataFrame a un nuevo archivo CSV
new_df.to_csv('mati_aumentado.csv', index=False)

print("Datos aumentados y guardados en 'ruta/a/tu/archivo_aumentado.csv'")
