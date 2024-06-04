import csv
import re

# Ruta del archivo de texto
input_file_path = 'chat.csv'
output_file_path = 'chat1.csv'

# Inicializar listas para las partes del mensaje
dates = []
names = []
messages = []

# Leer el archivo de texto
with open(input_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        # Extraer la fecha, el nombre y el mensaje utilizando expresiones regulares
        match = re.match(r'\[(\d+/\d+/\d+, \d+:\d+:\d+)\] (.*?): (.*)', line)
        if match:
            date = match.group(1)
            name = match.group(2)
            message = match.group(3)
            dates.append(date)
            names.append(name)
            messages.append(message)

# Escribir las partes del mensaje en un archivo CSV con separador '|'
with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter='|')
    # Escribir la cabecera
    csvwriter.writerow(['Fecha', 'Nombre', 'Mensaje'])
    # Escribir los datos
    for date, name, message in zip(dates, names, messages):
        csvwriter.writerow([date, name, message])

print(f"Archivo guardado en {output_file_path}")
