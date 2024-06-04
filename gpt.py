from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AdamW
from torch.utils.data import Dataset, DataLoader
import streamlit as st
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import csv

# Cargar el tokenizer y el modelo entrenado
tokenizer = AutoTokenizer.from_pretrained("datificate/gpt2-small-spanish")
model = GPT2LMHeadModel.from_pretrained("fine_tuned_gpt2")

# Función para asegurar que las respuestas sean oraciones completas
def ensure_complete_sentences(text):
    if '.' in text:
        return text[:text.rfind('.') + 1]
    return text

# Función para autocompletar texto y generar múltiples opciones
def autocomplete_text(prompt_text, num_responses=3, max_length=200, temperature=0.4, top_k=50):
    # Codificar el texto de entrada en tensores
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

    # Generar múltiples textos con el modelo
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=num_responses,
        do_sample=True,  # Usar búsqueda estocástica
    )

    # Decodificar los textos generados y asegurar que no excedan la longitud máxima
    completed_texts = []
    for output in outputs:
        decoded_text = tokenizer.decode(output, skip_special_tokens=True)
        # Limitar la longitud del texto generado y asegurar que sean oraciones completas
        truncated_text = ensure_complete_sentences(decoded_text[:max_length])
        completed_texts.append(truncated_text)

    return completed_texts

# Función principal para obtener la entrada del usuario y mostrar las opciones de autocompletado
def main():
    st.title("Generador de Texto con GPT-2")

    # Estado persistente para mantener el texto generado y las opciones
    if 'generated_text' not in st.session_state:
        st.session_state.generated_text = ""
    if 'response_options' not in st.session_state:
        st.session_state.response_options = []
    if 'max_length' not in st.session_state:
        st.session_state.max_length = 200
    if 'rows_processed' not in st.session_state:
        st.session_state.rows_processed = 0

    # Campo de texto para entrada del usuario
    prompt_text = st.text_input("Escriba el texto inicial o continúe el texto generado:")

    # Botón para generar opciones
    if st.button("Generar opciones"):
        # Usar el texto acumulado como el nuevo prompt
        full_prompt = st.session_state.generated_text + " " + prompt_text
        response_options = autocomplete_text(full_prompt, num_responses=3, max_length=st.session_state.max_length)

        st.session_state.response_options = response_options
        st.session_state.prompt_text = prompt_text  # Guardar el texto del prompt actual
        st.session_state.max_length *= 2  # Duplicar el max_length

    # Mostrar opciones generadas si están disponibles
    if st.session_state.response_options:
        st.write("Opciones generadas:")
        for i, response in enumerate(st.session_state.response_options, 1):
            st.write(f"Opción {i}: {response}")

        # Elegir una opción
        choice = st.radio("Elija una opción:", [f"Opción {i}" for i in range(1, len(st.session_state.response_options) + 1)], index=0)

        # Esperar a que el usuario haga una selección antes de actualizar el texto generado
        if st.button("Confirmar selección"):
            # Obtener la opción seleccionada y actualizar el texto generado
            selected_option = int(choice.split()[1]) - 1
            st.session_state.generated_text += " " + st.session_state.response_options[selected_option]

            # Guardar la respuesta generada en un archivo CSV
            with open('generated_responses.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=['text'])
                writer.writerow({'text': st.session_state.generated_text})
                st.session_state.rows_processed += 1

                # Verificar si se han procesado 100 renglones y, en ese caso, entrenar el modelo
                if st.session_state.rows_processed % 100 == 0:
                    entrenamiento()

            # Mostrar el texto generado hasta ahora
            st.write("Texto generado hasta ahora:")
            st.write(st.session_state.generated_text)

            # Reiniciar las opciones y el texto ingresado por el usuario
            st.session_state.response_options = []
            st.session_state.prompt_text = ""

            # Limpiar el campo de texto inicial para permitir nuevas entradas
            st.experimental_rerun()

    else:
        # Mostrar el texto generado hasta ahora si no hay opciones disponibles
        st.write("Texto generado hasta ahora:")
        st.write(st.session_state.generated_text)


def entrenamiento():
    # Cargar los datos desde el archivo CSV
    data = pd.read_csv('generated_responses.csv')

    # Definir una clase de conjunto de datos personalizada para procesar los datos
    class CustomDataset(Dataset):
        def __init__(self, data, tokenizer, max_length=512):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            text = self.data.iloc[idx]['text']
            encoding = self.tokenizer.encode_plus(text, padding='max_length', max_length=self.max_length, return_tensors='pt', truncation=True)
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }

    # Crear una instancia del tokenizador y el modelo GPT-2 preentrenado
    tokenizer = GPT2Tokenizer.from_pretrained("datificate/gpt2-small-spanish")
    model = GPT2LMHeadModel.from_pretrained("datificate/gpt2-small-spanish")

    # Definir el conjunto de datos y el cargador de datos
    dataset = CustomDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Configurar el entrenamiento
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Entrenar el modelo
    model.train()
    for epoch in range(20):  # Realizar 3 épocas de entrenamiento
        for batch in tqdm(dataloader, desc=f'Epoch {epoch + 1}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Guardar el modelo entrenado
    model.save_pretrained("fine_tuned_gpt2")

    print("El modelo ha sido entrenado y guardado.")


if __name__ == "__main__":
    main()
