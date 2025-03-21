import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# Load dataset
data = pd.read_csv("train_data.csv")

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

# Preprocess ingredients column
def tokenize_text(text, tokenizer, max_length=128):
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    return tokens['input_ids'].squeeze(0), tokens['attention_mask'].squeeze(0)

# Custom dataset class
class EcoScoreDataset(Dataset):
    def __init__(self, dataframe, tokenizer, scaler=None):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        
        # Tokenize ingredients column
        self.inputs = [tokenize_text(str(text), tokenizer) for text in dataframe["ingredients"]]
        self.features = dataframe.drop(columns=["eco_score", "ingredients", "barcode", "name", "brand"], errors='ignore').values
        self.labels = dataframe["eco_score"].values.astype(float)
        
        # Normalize numerical features
        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        input_ids, attention_mask = self.inputs[idx]
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return input_ids, attention_mask, features, label

# Split dataset
train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)
scaler = StandardScaler()
scaler.fit(train_df.drop(columns=["eco_score", "ingredients", "barcode", "name", "brand"], errors='ignore').values)
train_dataset = EcoScoreDataset(train_df, tokenizer, scaler)
val_dataset = EcoScoreDataset(val_df, tokenizer, scaler)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Define Model
class EcoScoreModel(nn.Module):
    def __init__(self, bert_model, num_numeric_features):
        super(EcoScoreModel, self).__init__()
        self.bert = bert_model
        self.fc1 = nn.Linear(768 + num_numeric_features, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
    
    def forward(self, input_ids, attention_mask, numeric_features):
        with torch.no_grad():  # Freeze BERT during training
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embedding = bert_output.last_hidden_state[:, 0, :]
        combined_input = torch.cat((bert_embedding, numeric_features), dim=1)
        x = self.relu(self.fc1(combined_input))
        x = self.fc2(x)
        return x

# Initialize Model
num_numeric_features = train_dataset.features.shape[1]
model = EcoScoreModel(bert_model, num_numeric_features).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, features, labels in train_loader:
        input_ids, attention_mask, features, labels = input_ids.to(device), attention_mask.to(device), features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, features).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Save the model
torch.save(model.state_dict(), "bert_eco_score_model.pth")
print("✅ Fine-tuned BERT model saved as 'bert_eco_score_model.pth'")
