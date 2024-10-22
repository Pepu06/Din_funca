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

    response = response.candidates[0].content.parts[0].text

    return response

expected = """
- Hola, como estas?
- Hola, buen dia
- Hola, que tal?
"""

generated = generate_response(input("Ingrese un texto: "))
print("generated: ", generated)

comparacion = comparar_res(expected, generated)

print("comparacion: ", comparacion)