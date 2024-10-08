from flask import Flask, request, jsonify
import pandas as pd
import google.generativeai as genai
import speech_recognition as sr
from flask_cors import CORS
import csv
import pyttsx3
import time
import json

app = Flask(__name__)
CORS(app)
# Configura la clave de API de Gemini
GOOGLE_API_KEY = 'AIzaSyAI6SmUQbQ9wJohy53_kssfZuzoLG5FRes'
genai.configure(api_key=GOOGLE_API_KEY)

# Configura el modelo de generación de texto
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=genai.GenerationConfig(
        max_output_tokens=2000,
        temperature=1.2,
    ),
    system_instruction="Tienes que completar las oraciones que pasa el usuario de una forma que tenga sentido y sea concisa, incluyendo el texto que ingresa el usuario como el principio de la oración completa. Es importante que la respuesta sea únicamente la oración completa, sin agregar información adicional. No usar exactamente el texto que esta en el csv"
)

# Mantenemos el historial del chat en la sesión actual
chat_history = []

# Lee el archivo CSV y extrae el contenido de interés
csv_file = 'mati.csv'
df = pd.read_csv(csv_file, delimiter="|")
texts = df['text'].tolist()

@app.route('/')
def hello(): 
    return "Hola Dan"

# Función para buscar ejemplos relevantes basados en palabras clave
def buscar_ejemplos(palabras_clave, textos):
    ejemplos_relevantes = []
    for texto in textos:
        if any(palabra in texto for palabra in palabras_clave):
            ejemplos_relevantes.append(texto)
    return ejemplos_relevantes

# Función para ajustar la entrada al modelo basado en los ejemplos encontrados
def generate_prompt(audio_context, user_input, examples, history):
    prompt = "Contexto de la conversación:\n"
    prompt += f"- Contexto del micrófono: {audio_context}\n"
    for mensaje in history:
        prompt += f"- {mensaje}\n"
    prompt += f"\nIngresa el texto: {user_input}\n\n"
    prompt += "Basado en los siguientes ejemplos, completa la oración de manera similar:\n\n"
    for example in examples:
        prompt += f"- {example}\n"
    return prompt

# Función para completar el texto usando el contexto del audio y el input del usuario
@app.route('/autocompleteConAudio', methods=['POST'])
def completar_texto_con_audio():
    global chat_history
    audio_context = json.loads(request.get_data().decode("utf-8"))["audio"]
    user_input = json.loads(request.get_data().decode("utf-8"))["texto"]
    palabras_clave = user_input.split()
    ejemplos = buscar_ejemplos(palabras_clave, texts)
    if not ejemplos:
        ejemplos = texts[-5:]
    
    prompt = generate_prompt(audio_context, user_input, ejemplos, chat_history)
    prompt += "Para generar las respuestas utilizar el contexto de la conversacion pasada {audio_context}"

    respuestas = []
    tiempos_respuesta = []
    for _ in range(3):  # Generar tres respuestas
        start_time = time.time()
        response = model.generate_content([prompt])
        tiempo = time.time() - start_time
        tiempos_respuesta.append(tiempo)
        respuesta_texto = response.text.strip()
        respuestas.append(respuesta_texto)

    # Mostrar tiempos de respuesta
    print(f"Tiempos de respuesta (segundos): {tiempos_respuesta}")

    # Actualizar el historial del chat
    chat_history.append(f"Contexto del micrófono: {audio_context}")
    chat_history.append(f"Usuario: {user_input}")
    for resp in respuestas:
        chat_history.append(f"AI: {resp}")

    # Limitar el tamaño del historial para no acumular demasiado
    if len(chat_history) > 20:
        chat_history = chat_history[-20:]

    return respuestas

# Función para completar el texto solo con el input del usuario
@app.route('/autocomplete', methods=['POST'])
def completar_texto_con_usuario():
    global chat_history
    user_input = json.loads(request.get_data().decode("utf-8"))["texto"]
    palabras_clave = user_input.split()
    ejemplos = buscar_ejemplos(palabras_clave, texts)
    if not ejemplos:
        ejemplos = texts[-5:]
    
    prompt = f"Ingresa el texto: {user_input}\n\n"
    prompt += "Basado en los siguientes ejemplos, completa la oración de manera similar:\n\n"
    for example in ejemplos:
        prompt += f"- {example}\n"

    respuestas = []
    tiempos_respuesta = []
    for _ in range(3):  # Generar tres respuestas
        start_time = time.time()
        response = model.generate_content([prompt])
        tiempo = time.time() - start_time
        tiempos_respuesta.append(tiempo)
        respuesta_texto = response.text.strip()
        respuestas.append(respuesta_texto)

    # Mostrar tiempos de respuesta
    print(f"Tiempos de respuesta (segundos): {tiempos_respuesta}")

    # Actualizar el historial del chat
    chat_history.append(f"Usuario: {user_input}")
    for resp in respuestas:
        chat_history.append(f"AI: {resp}")

    # Limitar el tamaño del historial para no acumular demasiado
    if len(chat_history) > 20:
        chat_history = chat_history[-20:]

    return respuestas

# Función para escuchar el micrófono y convertir el audio a texto
def escuchar_mic():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Escuchando...")
        audio = recognizer.listen(source)
        try:
            texto = recognizer.recognize_google(audio, language="es-ES")
            print(f"Texto reconocido: {texto}")
            return texto
        except sr.UnknownValueError:
            print("No se pudo reconocer el audio.")
            return None
        except sr.RequestError as e:
            print(f"Error en la solicitud de reconocimiento: {e}")
            return None

# Función para agregar texto a un archivo CSV
def agregar_texto_a_csv(archivo_csv, texto):
    with open(archivo_csv, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='|')
        writer.writerow([texto])

# Función para leer texto en voz alta
def lee_texto(texto):
    engine = pyttsx3.init()
    engine.save_to_file(texto, 'hello.mp3')
    engine.runAndWait()
    return 'hello.mp3'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
