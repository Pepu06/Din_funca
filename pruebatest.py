from generatyresponsychan import generate_response

# Función ficticia para simular la interacción con gemini-pro
def gemini_similarity(text1, text2):
    # Implementa aquí la llamada al modelo gemini-pro
    # Por ejemplo:
    # return gemini_model.compare(text1, text2)  # Reemplaza con tu implementación
    # Simulación simple
    return 0.85  # Simular un 85% de similitud

def compare_responses(generated_responses, expected_responses):
    correct_count = 0
    threshold = 0.75  # Umbral de similitud

    # Dividir la respuesta generada en partes
    generated_responses_list = generated_responses.split("\n")  # Asumiendo que las respuestas están separadas por saltos de línea

    # Asegurarse de que las listas tienen la misma longitud
    min_length = min(len(generated_responses_list), len(expected_responses))
    
    for generated, expected in zip(generated_responses_list[:min_length], expected_responses[:min_length]):
        # Calcular la similitud usando gemini-pro
        similarity = gemini_similarity(generated, expected)
        
        # Si la similitud es mayor que el umbral, contar como correcta
        if similarity >= threshold:
            correct_count += 1
            
    return (correct_count / min_length) * 100  # Calcular porcentaje de efectividad


# Ejemplo de uso
if __name__ == "__main__":
    user_text = "Es un día soleado."  # Texto del usuario
    expected_responses = [
        "Es un día soleado. Sería ideal salir a caminar.",
        "Es un día soleado. Tal vez podríamos ir al parque.",
        "Es un día soleado. Perfecto para una barbacoa."
    ]
    
    # Generar respuestas
    generated_responses = generate_response(user_text)  # Asegúrate de que esto retorne una única cadena con tres respuestas

    # Comparar respuestas
    percentage_effectiveness = compare_responses(generated_responses, expected_responses)
    
    print(f"Porcentaje de efectividad: {percentage_effectiveness:.2f}%")
