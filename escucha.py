import pandas as pd
import google.generativeai as genai
import speech_recognition as sr

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
csv_file = '/mati.csv'
df = pd.read_csv(csv_file, delimiter="|")
texts = df['text'].tolist()

# Función para buscar ejemplos relevantes basados en palabras clave
def buscar_ejemplos(palabras_clave, textos):
    ejemplos_relevantes = []
    for texto in textos:
        if any(palabra in texto for palabra in palabras_clave):
            ejemplos_relevantes.append(texto)
    return ejemplos_relevantes

# Función para ajustar la entrada al modelo basado en los ejemplos encontrados
def generate_prompt(user_input, examples, history):
    prompt = "Contexto de la conversación:\n"
    for mensaje in history:
        prompt += f"- {mensaje}\n"
    prompt += f"\nIngresa el texto: {user_input}\n\n"
    prompt += "Basado en los siguientes ejemplos, completa la oración de manera similar:\n\n"
    for example in examples:
        prompt += f"- {example}\n"
    return prompt

# Función para completar el texto usando Gemini y obtener múltiples respuestas
def completar_texto(texto_usuario):
    global chat_history
    palabras_clave = texto_usuario.split()
    ejemplos = buscar_ejemplos(palabras_clave, texts)
    if not ejemplos:
        ejemplos = texts[-5:]
    prompt = generate_prompt(texto_usuario, ejemplos, chat_history)

    respuestas = []
    for _ in range(3):  # Generar tres respuestas
        response = model.generate_content([prompt])
        respuesta_texto = response.text.strip()
        respuestas.append(respuesta_texto)

    # Actualizar el historial del chat
    chat_history.append(f"Usuario: {texto_usuario}")
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

# Interacción con el usuario
def main():
    global chat_history
    while True:
        print("Di algo para usar como contexto (o di 'salir' para terminar):")
        contexto_audio = escuchar_mic()
        if contexto_audio is None:
            continue
        if contexto_audio.lower() == 'salir':
            break

        print(f"Contexto capturado: {contexto_audio}")
        chat_history.append(f"Contexto: {contexto_audio}")

        while True:
            user_input = input("Ingresa un texto (o 'salir' para terminar): ")
            if user_input.lower() == 'salir':
                break
            respuestas = completar_texto(user_input)
            for idx, respuesta in enumerate(respuestas, start=1):
                print(f"Respuesta {idx}: {respuesta}")

if __name__ == '__main__':
    main()
