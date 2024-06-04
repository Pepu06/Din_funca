import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from tqdm import tqdm

# Cargar los datos desde el archivo CSV
file_path = 'chat1.csv'  # Asegúrate de proporcionar la ruta correcta
data = pd.read_csv(file_path, delimiter='|')

# Definir una clase de conjunto de datos personalizada para procesar los datos
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['Mensaje']
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
for epoch in range(10):  # Realizar 3 épocas de entrenamiento
    for batch in tqdm(dataloader, desc=f'Epoch {epoch + 1}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Guardar el modelo entrenado
model.save_pretrained("fine_tuned_gpt2_pedro")

print("El modelo ha sido entrenado y guardado.")
