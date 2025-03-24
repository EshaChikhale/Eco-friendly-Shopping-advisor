import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
import pandas as pd
from tqdm import tqdm
import os

# ✅ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ✅ Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ✅ Define Custom Dataset
class FoodDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]  # ✅ Return raw text and label

# ✅ Custom collate function for dynamic padding
def collate_fn(batch):
    texts, labels = zip(*batch)  # ✅ Unpack batch data
    encoded = tokenizer(list(texts), padding=True, truncation=True, max_length=64, return_tensors="pt")  
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long),
    }

if __name__ == "__main__":
    # ✅ Load Dataset
    dataset_path = "en.openfoodfacts.org.products.tsv"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ Dataset file not found: {dataset_path}")

    print("📂 Loading dataset...")
    df = pd.read_csv(dataset_path, sep='\t', low_memory=False)

    # ✅ Drop NaN values and keep relevant columns
    df = df[['product_name', 'categories']].dropna()

    # ✅ Keep only top 500 most frequent categories, label others as "Other"
    top_categories = df['categories'].value_counts().index[:500]
    df['categories'] = df['categories'].apply(lambda x: x if x in top_categories else "Other")

    # ✅ Encode categories
    category_codes, unique_categories = pd.factorize(df['categories'])
    df['category'] = category_codes

    num_labels = len(unique_categories)
    print(f"✅ Reduced categories to {num_labels}")

    # ✅ Reduce dataset size for faster training
    df = df.sample(50000, random_state=42)

    # ✅ Prepare dataset & dataloaders
    dataset = FoodDataset(df['product_name'].tolist(), df['category'].tolist())
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn)

    # ✅ Load Model (`BERT-Base` for faster training)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels).to(device)

    # ✅ Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, eps=1e-8, weight_decay=0.01)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=800, num_training_steps=len(train_loader) * 5)
    loss_fn = nn.CrossEntropyLoss()

    # ✅ Mixed Precision Training for Speed
    scaler = torch.amp.GradScaler("cuda")

    # ✅ Training Loop (5 Epochs)
    epochs = 5
    print("🚀 Training started...")
    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda"):
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            total_loss += loss.item()
            correct += (outputs.logits.argmax(dim=1) == labels).sum().item()

        train_acc = (correct / train_size) * 100
        print(f"✅ Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Accuracy: {train_acc:.2f}%")

    # ✅ Evaluation
    print("🧪 Evaluating model...")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(input_ids, attention_mask=attention_mask)
            correct += (outputs.logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    test_acc = (correct / total) * 100
    print(f"🎯 Test Accuracy: {test_acc:.2f}%")

    # ✅ Save the trained model
    model.save_pretrained("bert_base_food_classifier")
    tokenizer.save_pretrained("bert_base_food_classifier")
    print("✅ Model saved successfully!")

