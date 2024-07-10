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

    # Genera el prompt con ejemplos (usamos menos ejemplos para reducir el tiempo de respuesta)
    prompt = generate_prompt(user_text, texts[-5:])  # Usamos los últimos 5 ejemplos
    # prompt = generate_prompt(user_text, texts)  # Usamos TODOS LOS EJEMPLOS

    # Función para generar y recolectar respuestas
    def generate_responses(prompt, n=3):
        responses = []
        for _ in range(n):
            response = ""
            print(end=' ', flush=True)
            for chunk in llm.stream(prompt):
                for char in chunk:
                    print(char, end='', flush=True)
                    response += char
            responses.append(response)
            print("\n\n")  # Nueva línea al final de cada respuesta
        return responses

    # Genera tres respuestas
    responses = generate_responses(prompt, n=3)
    return responses


# Prueba de la función
autocompletado(input("Ingresa el texto: "))