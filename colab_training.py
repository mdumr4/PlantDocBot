# ==========================================
# PLANT DOC BOT - TRAINING SCRIPT (COLAB)
# ==========================================
# UPDATED WITH DETAILED EVALUATION METRICS
#
# INSTRUCTIONS:
# 1. Open Google Colab (https://colab.research.google.com/)
# 2. Select T4 GPU Runtime.
# 3. Copy/Paste this script.
# 4. Run.
# ==========================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import opendatasets as od
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import numpy as np

# ==========================================
# 1. CONFIGURATION
# ==========================================
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
DATASET_URL = "https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset"
MODEL_SAVE_PATH = "plant_doctor_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. DATA PREPROCESSING & DOWNLOAD
# ==========================================
if not os.path.exists("./new-plant-diseases-dataset"):
    print("\n--- Downloading Dataset ---")
    od.download(DATASET_URL)

data_dir = "./new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_dataset.classes
with open("class_names.txt", "w") as f:
    for c in class_names:
        f.write(c + "\n")

# ==========================================
# 3. MODEL SETUP (ResNet34)
# ==========================================
model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 4. TRAINING WITH METRICS
# ==========================================
print("\n--- Starting Training ---")

train_losses, valid_losses = [], []
train_accs, valid_accs = [], []

for epoch in range(EPOCHS):
    # --- TRAINING PHASE ---
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_train_loss = running_loss / len(train_loader)
    epoch_train_acc = 100 * correct / total
    train_losses.append(epoch_train_loss)
    train_accs.append(epoch_train_acc)

    # --- VALIDATION PHASE (WITH DETAILED METRICS) ---
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_val_loss = val_loss / len(valid_loader)
    valid_losses.append(epoch_val_loss)

    # Calculate Sklearn Metrics
    val_acc = accuracy_score(all_labels, all_preds) * 100
    valid_accs.append(val_acc)

    # Macro Average: good for treating all classes equally regardless of size
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
    print(f"  Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    print(f"  Val Metrics: Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")

# ==========================================
# 5. FINAL EVALUATION REPORT
# ==========================================
print("\n--- Final Evaluation Report ---")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Save Plots
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(valid_accs, label='Valid Acc')
plt.legend()
plt.title('Accuracy Curve')
plt.savefig('training_plots.png')
print("Plots saved as training_plots.png")

# Save Model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
