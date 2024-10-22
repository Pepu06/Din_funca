import os
from dotenv import load_dotenv
import google.generativeai as genai
from generatyresponsychan import generate_response
from generatyresponsychan import load_chroma_collection

# Load environment variables
load_dotenv()

# Set up your Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-1.5-pro")


def comparar_res(texto1, texto2):
    # Crear un prompt para hacer la comparación entre los dos textos
    prompt = f"Compara la similitud entre los siguientes dos textos:\n\nTexto 1: {texto1}\n\nTexto 2: {texto2}\n\n" \
             f"Unicamente indica qué tan similares son en una escala del 1 al 100"

    response = model.generate_content(prompt)

    return response.text

texto1 = """
- Hola, como estas?
- Hola, buen dia
- Hola, que tal?
"""

texto2 = generate_response(input("Ingrese un texto: "))
print(texto2)

comparacion = comparar_res(texto1, texto2)

print(comparacion)