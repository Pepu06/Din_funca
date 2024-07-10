import tkinter as tk
from tkinter import ttk
import pandas as pd
from langchain_community.llms import Ollama

def autocompletado(texto):
    # Inicializa el modelo de Ollama
    llm = Ollama(
        model="llama3:latest",
        system="Tienes que completar las oraciones que pasa el usuario de una forma que tenga sentido y sea concisa, incluyendo el texto que ingresa el usuario como el principio de la oracion completa. Es importante que la repuesta sea unicamente la oración completa, sin agregar información adicional."
    )

    # Lee el archivo CSV y extrae el contenido de interés
    csv_file = 'mati.csv'  # Reemplaza con la ruta a tu archivo CSV
    df = pd.read_csv(csv_file)

    # Supongamos que la columna 'text' contiene los textos de interés
    texts = df['text'].tolist()

    # Función para ajustar la entrada al modelo basado en los textos del CSV
    def generate_prompt(user_input, examples):
        prompt = f"Ingresa el texto: {user_input}\n\n"
        prompt += "Basado en los siguientes ejemplos, completa la oración de manera similar:\n\n"
        for example in examples:
            prompt += f"- {example}\n"
        return prompt

    # Obtiene el texto del usuario
    user_text = texto

    # Genera el prompt con todos los ejemplos
    prompt = generate_prompt(user_text, texts)  # Usamos todos los ejemplos

    # Función para generar y recolectar respuestas
    def generate_responses(prompt, n=3):
        responses = []
        for _ in range(n):
            response = ""
            for chunk in llm.stream(prompt):
                for char in chunk:
                    response += char
            responses.append(response)
        return responses

    # Genera tres respuestas
    responses = generate_responses(prompt, n=3)
    return responses

# Función de actualización cuando el usuario presiona espacio
def on_space_press(event):
    if event.char == ' ':
        user_text = text_var.get()
        responses = autocompletado(user_text)
        for i, response in enumerate(responses):
            response_vars[i].set(response)

# Configuración de la interfaz gráfica con tkinter
root = tk.Tk()
root.title("Autocompletado Dinámico")

mainframe = ttk.Frame(root, padding="10 10 10 10")
mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

text_var = tk.StringVar()
response_vars = [tk.StringVar() for _ in range(3)]

ttk.Label(mainframe, text="Ingresa el texto:").grid(column=1, row=1, sticky=tk.W)
text_entry = ttk.Entry(mainframe, width=50, textvariable=text_var)
text_entry.grid(column=2, row=1, sticky=(tk.W, tk.E))
text_entry.bind('<KeyRelease>', on_space_press)

for i in range(3):
    ttk.Label(mainframe, textvariable=response_vars[i]).grid(column=1, row=2+i, columnspan=2, sticky=(tk.W, tk.E))

for child in mainframe.winfo_children(): 
    child.grid_configure(padx=5, pady=5)

root.mainloop()
