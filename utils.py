
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Constants
CLASS_NAMES_FILE = "data/class_names.txt"
DISEASE_INFO_FILE = "data/disease_info.json"
VISUAL_MODEL_FILE = "models/plant_doctor_model.pth"
BERT_MODEL_DIR = "models/plant_disease_bert"
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

@st.cache_resource
def load_models_and_data():
    """
    Loads all necessary resources:
    1. Class Names
    2. Disease Info
    3. Visual Model (ResNet34)
    4. Text Model (BERT or SBERT)
    5. Knowledge Base Embeddings (if SBERT)
    """
    error_msg = None

    # 1. Load Class Names
    if not os.path.exists(CLASS_NAMES_FILE):
        return None, None, None, None, None, None, "MISSING_CLASSES"

    with open(CLASS_NAMES_FILE, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # 2. Load Disease Info
    disease_info = {}
    if os.path.exists(DISEASE_INFO_FILE):
        with open(DISEASE_INFO_FILE, "r") as f:
            disease_info = json.load(f)

    # 3. Load Visual Model (ResNet34)
    model_viz = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(VISUAL_MODEL_FILE):
        try:
            model_viz = models.resnet34(weights=None)
            num_ftrs = model_viz.fc.in_features
            model_viz.fc = nn.Linear(num_ftrs, len(class_names))
            model_viz.load_state_dict(torch.load(VISUAL_MODEL_FILE, map_location=device))
            model_viz.to(device)
            model_viz.eval()
        except Exception as e:
            error_msg = f"VISUAL_MODEL_ERROR: {str(e)}"
    else:
        error_msg = "MISSING_MODEL"

    # 4. Load Text Model
    model_text = None
    bert_pipe = None

    try:
        if os.path.exists(BERT_MODEL_DIR):
            bert_pipe = pipeline("text-classification", model=BERT_MODEL_DIR)
        else:
            model_text = SentenceTransformer(SBERT_MODEL_NAME)
    except Exception as e:
         return None, None, None, None, None, None, f"TEXT_MODEL_ERROR: {str(e)}"

    # 5. Pre-compute Text Embeddings (SBERT only)
    kb_embeddings = None
    if model_text:
        symptom_texts = []
        for cls in class_names:
            info = disease_info.get(cls, {})
            text = info.get('symptoms') or info.get('description') or cls.replace("___", " ")
            symptom_texts.append(text)
        kb_embeddings = model_text.encode(symptom_texts, convert_to_tensor=True)

    status = "SUCCESS" if not error_msg else error_msg
    return model_viz, model_text, bert_pipe, class_names, disease_info, kb_embeddings, status

def process_image(image):
    """Preprocesses image for ResNet34"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict_image(model, image, class_names, device=None):
    """
    Runs inference on an image.
    Returns: (predicted_class_name, confidence_score, probabilities_array)
    """
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_tensor = process_image(image).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, 1)

    return class_names[idx.item()], conf.item() * 100, probs.cpu().numpy()[0]

def predict_text_bert(bert_pipe, text, class_names):
    """
    Runs inference using BERT pipeline.
    Returns: (predicted_class_name, confidence_score, label_index)
    """
    pred = bert_pipe(text)[0]
    # Label is usually "LABEL_X" or "Class_X"
    label_str = pred['label']
    idx_str = "".join(filter(str.isdigit, label_str))
    idx = int(idx_str)

    return class_names[idx], pred['score'] * 100, idx

def predict_text_sbert(model_text, kb_embeddings, text, class_names):
    """
    Runs semantic search using SBERT.
    Returns: (predicted_class_name, confidence_score, scores_array, top_indices)
    """
    user_embedding = model_text.encode(text, convert_to_tensor=True)
    cos_scores = util.cos_sim(user_embedding, kb_embeddings)[0]

    top_results = torch.topk(cos_scores, k=len(class_names))
    best_idx = top_results.indices[0].item()
    best_score = top_results.values[0].item() * 100

    return class_names[best_idx], best_score, cos_scores.cpu().numpy(), top_results
