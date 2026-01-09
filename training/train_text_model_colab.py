
# ==========================================
# PLANT DOC BOT - TEXT MODEL TRAINING (COLAB)
# ==========================================
#
# INSTRUCTIONS:
# 1. Open Google Colab: https://colab.research.google.com/
# 2. Runtime -> Change runtime type -> T4 GPU
# 3. Copy AND Paste this entire script into a cell.
# 4. Run the cell.
# 5. When done, a folder 'plant_disease_bert' will be created.
# 6. Zip and Download that folder!
# ==========================================

import os

# --- 1. INSTALL DEPENDENCIES ---
print("Installing transformers and datasets...")
os.system("pip install transformers datasets scikit-learn accelerate -U")

# Disable W&B logging to avoid login prompts
os.environ["WANDB_DISABLED"] = "true"

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- 2. CONFIGURATION ---
MODEL_NAME = "distilbert-base-uncased" # Fast and good enough
DATASET_ID = "ButterChicken98/plantvillage-image-text-pairs"
OUTPUT_DIR = "./plant_disease_bert"
NUM_EPOCHS = 3
BATCH_SIZE = 16

# --- 3. PREPARE DATA ---
print(f"Downloading dataset: {DATASET_ID}...")
dataset = load_dataset(DATASET_ID, split="train")

# We need to GENERATE labels from the text/caption because the dataset doesn't have them
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

print("Synthesizing Labels from Captions...")
def assign_label(example):
    # Try to find which class name is inside the caption
    # caption structure is usually somewhat descriptive
    text = (example.get('caption') or example.get('text') or "").lower()

    # We strip special chars for matching: "Apple___Apple_scab" -> "apple apple scab"
    for idx, cls_name in enumerate(CLASS_NAMES):
        # Create a simplified version of class name
        clean_cls = cls_name.lower().replace("___", " ").replace("_", " ").replace("(", "").replace(")", "")
        # Also try strict raw match if clean fails

        # Heuristic: Check if the key parts are in text
        # Simplest: Just check if clean_cls is substing of text
        # (This works if dataset captions are generated from these labels)
        if clean_cls in text:
            return {'labels': idx}

        # Fallback for complex names?
        # Maybe "corn common rust" matches "Corn_(maize)___Common_rust_"
        parts = clean_cls.split()
        if len(parts) > 2 and all(p in text for p in parts):
             return {'labels': idx}

    return {'labels': -1} # Unlabeled

dataset = dataset.map(assign_label)
# Filter out Unlabeled
dataset = dataset.filter(lambda x: x['labels'] != -1)
print(f"Filtered Dataset Size: {len(dataset)}")

# Split into Train/Test (80/20)
dataset = dataset.train_test_split(test_size=0.2)
# Determine text column for tokenizer
text_col = "caption" if "caption" in dataset["train"].column_names else "text"
print(f"Using text column: {text_col}")

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    # Use detected column
    texts = [t if t else "" for t in examples[text_col]]
    return tokenizer(texts, truncation=True, padding=False) # Padding handled by collator

print("Tokenizing data...")
tokenized_datasets = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --- 4. MODEL SETUP ---
# Determine number of classes (assume 38 for PlantVillage)
num_labels = 38
# If dataset has fewer/more, this might crash.
# Ideally we count unique labels:
# num_labels = len(set(dataset['train']['label']))

print(f"Loading Model (Num Labels: {num_labels})...")
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label={i: f"Class_{i}" for i in range(num_labels)}, # Placeholder names
    label2id={f"Class_{i}": i for i in range(num_labels)}
)

# --- 5. TRAINING ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\n--- STARTING TRAINING ---")
trainer.train()

# --- 6. SAVE ARTIFACTS ---
print(f"\n--- SAVING MODEL TO {OUTPUT_DIR} ---")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nDONE! Now zip and download the 'plant_disease_bert' folder.")
print("Command to zip: !zip -r plant_disease_bert.zip plant_disease_bert")
