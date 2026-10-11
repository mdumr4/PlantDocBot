# 🌿 PlantDocBot - Multimodal AI Plant Disease Diagnosis

<p align="center">
  <em>An enterprise-grade agricultural intelligence platform combining deep convolutional vision (ResNet34) and fine-tuned natural language processing (DistilBERT) for real-time crop disease diagnosis and treatment recommendation.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Torchvision-ResNet34-00599C?style=for-the-badge" alt="ResNet34">
  <img src="https://img.shields.io/badge/HuggingFace-DistilBERT-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
  <img src="https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Infosys-Springboard_Certified-0070AD?style=for-the-badge" alt="Infosys Springboard">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License"></a>
</p>

---

## Executive Summary

Agricultural crop diseases inflict over $220 billion in annual global harvest losses. Smallholder farmers and agriculturalists often face severe diagnostic delays due to limited on-site access to plant pathologists, leading to crop loss or indiscriminate chemical application.

Developed as an **Infosys Springboard** AI milestone project, **PlantDocBot** solves this through a multimodal diagnostic architecture:
- **Computer Vision (ResNet34)**: Identifies 38 distinct crop diseases with 95.2%+ validation accuracy on high-resolution leaf imagery.
- **Natural Language Processing (DistilBERT)**: Interprets free-form symptom descriptions and handles contextual agricultural nuance.
- **Hybrid Probabilistic Blending**: Fuses visual logits ($70\%$) and textual symptom probabilities ($30\%$) to resolve diagnostic ambiguities caused by leaf glare or partial occlusions.
- **Curated Knowledge Base**: Maps diagnoses to actionable organic remedies, chemical treatments, and preventative protocols.

---

## System Architecture

```mermaid
flowchart TD
    subgraph Inputs["1. Multimodal Diagnostic Inputs"]
        LeafImg["Leaf Photograph (Upload: JPG/PNG)"]
        SymptomQuery["Natural Language Symptom Description"]
    end

    subgraph VisionTower["2. Computer Vision Pipeline (ResNet34)"]
        Preprocess["Resize 224x224 & ImageNet Normalization"]
        ResNet["Deep CNN Feature Extractor (ResNet34)"]
        VisLogits["Visual Probability Distribution (Pv)"]
    end

    subgraph TextTower["3. NLP Diagnostic Pipeline (DistilBERT / SBERT)"]
        Tokenizer["Hugging Face Tokenizer"]
        DistilBERT["Fine-Tuned DistilBERT Classifier"]
        SBERTSearch["SBERT (all-MiniLM-L6-v2) Fallback"]
        TextLogits["Textual Probability Distribution (Pt)"]
    end

    subgraph FusionEngine["4. Multimodal Fusion & Decision Engine"]
        Fusion["Weighted Hybrid Blend: 0.7*Pv + 0.3*Pt"]
        ArgMax["Top-1 Calibrated Class Identification"]
    end

    subgraph Knowledge["5. Recommendation & Treatment Engine"]
        KB[("data/disease_info.json (38 Classes)")]
        Prescription["Actionable Treatment & Prevention Protocol"]
    end

    subgraph Frontend["6. Streamlit Web Interface"]
        UI["Interactive Diagnosis Card & Confidence Meter"]
    end

    LeafImg --> Preprocess --> ResNet --> VisLogits
    SymptomQuery --> Tokenizer --> DistilBERT --> TextLogits
    SymptomQuery -.->|Fallback if no weights| SBERTSearch -.-> TextLogits
    VisLogits & TextLogits --> Fusion --> ArgMax
    ArgMax --> Prescription
    KB --> Prescription
    Prescription --> UI
```

---

## Interaction Workflow

```mermaid
sequenceDiagram
    autonumber
    actor Farmer as User / Farmer
    participant UI as Streamlit Web App (app.py)
    participant Core as Inference Engine (utils.py)
    participant Vision as ResNet34 CNN
    participant NLP as Fine-Tuned DistilBERT
    participant KB as Knowledge Base (disease_info.json)

    Farmer->>UI: Selects Tab (Image Only, Text Only, or Hybrid)
    Farmer->>UI: Uploads Leaf Image + Types Symptom Query
    UI->>Core: Pass Inputs to predict_image() & predict_text_bert()
    Core->>Vision: Forward Pass (224x224 Normalized Tensor)
    Vision-->>Core: Visual Class Probabilities (Pv)
    Core->>NLP: Tokenize & Forward Pass (DistilBERT)
    NLP-->>Core: Text Class Probabilities (Pt)
    Core->>Core: Calculate Weighted Fusion: (0.7 * Pv) + (0.3 * Pt)
    Core->>KB: Query Top-1 Disease Entry
    KB-->>Core: Disease Description, Treatment & Symptoms
    Core-->>UI: Output Class Name, Confidence %, & Treatment Card
    UI-->>Farmer: Renders Diagnosis & Prescription
```

---

## Benchmark & Model Performance Profile

| Model Component | Architecture / Backbone | Dataset | Training Specs | Key Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Vision Model** | **ResNet34 Deep CNN** | PlantVillage (50,000+ images) | 5 Epochs, Batch 32, Adam (lr=1e-3) | **95.2% Accuracy** |
| **Text Model** | **DistilBERT (distilbert-base-uncased)** | `ButterChicken98/plantvillage-image-text-pairs` | 3 Epochs, T4 GPU, HF Trainer | **91.8% Accuracy (F1: 0.91)** |
| **Text Fallback** | **Sentence-Transformers (all-MiniLM-L6-v2)** | Curated Symptom Corpus (38 classes) | Zero-shot cosine semantic similarity | **Top-3 Recall: 89.4%** |
| **Hybrid Mode** | **Probabilistic Blend ($70\% \text{ Vis} + 30\% \text{ Text}$)** | Multimodal Evaluation Pairs | Joint decision fusion | **96.8% Accuracy** |

---

## Supported Crops & Disease Coverage

PlantDocBot supports **38 distinct health and disease classes** across **14 major agricultural crops**:

| Crop Family | Conditions & Diagnoses Covered |
| :--- | :--- |
| 🍏 **Apple** | Apple scab, Black rot, Cedar apple rust, Healthy |
| 🫐 **Blueberry** | Healthy |
| 🍒 **Cherry** | Powdery mildew, Healthy |
| 🌽 **Corn (Maize)** | Cercospora leaf spot (Gray leaf spot), Common rust, Northern Leaf Blight, Healthy |
| 🍇 **Grape** | Black rot, Esca (Black Measles), Leaf blight (Isariopsis), Healthy |
| 🍊 **Orange** | Huanglongbing (Citrus greening) |
| 🍑 **Peach** | Bacterial spot, Healthy |
| 🫑 **Bell Pepper** | Bacterial spot, Healthy |
| 🥔 **Potato** | Early blight, Late blight, Healthy |
| 🍇 **Raspberry** | Healthy |
| 🌱 **Soybean** | Healthy |
| 🎃 **Squash** | Powdery mildew |
| 🍓 **Strawberry** | Leaf scorch, Healthy |
| 🍅 **Tomato** | Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Tomato Yellow Leaf Curl Virus, Tomato mosaic virus, Healthy |

---

## Quickstart (3 Steps)

### 1. Clone & Environment Setup
```bash
git clone https://github.com/mdumr4/Infosys-PlantBot.git
cd Infosys-PlantBot

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Weights Setup
Because deep learning checkpoints are large, model files are excluded from git. You have two options:

- **Option A: Download Pre-Trained Weights (Instant)**
  - Download the pre-trained weights from [Google Drive](https://drive.google.com/drive/folders/1TBWM30fXpK_mGARJQryr7iU2HtItPa1X?usp=sharing).
  - Extract `plant_doctor_model.pth` and the `plant_disease_bert/` folder into the `models/` directory:
    ```text
    models/
    ├── plant_doctor_model.pth
    └── plant_disease_bert/
        ├── model.safetensors
        ├── config.json
        └── tokenizer_config.json
    ```

- **Option B: Run with Zero-Shot Fallback**
  - If no weights are placed in `models/`, the application automatically falls back to Sentence-Transformers (`all-MiniLM-L6-v2`) for semantic symptom search.

### 3. Launch Streamlit Web Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Model Training & Colab Runbooks

To retrain the models from scratch on GPU infrastructure:

### 1. Training the Visual Model (ResNet34)
- **Script**: `training/train_visual_model.py`
- **Environment**: Google Colab (T4 GPU recommended)
- **Dataset**: [PlantVillage Kaggle Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Outputs**: Generates `plant_doctor_model.pth` (save into `models/`).

### 2. Training the Text Model (DistilBERT)
- **Script**: `training/train_text_model.py`
- **Environment**: Google Colab (T4 GPU recommended)
- **Dataset**: `ButterChicken98/plantvillage-image-text-pairs` (Hugging Face)
- **Outputs**: Saves fine-tuned checkpoint folder `plant_disease_bert/` (save into `models/`).

---

## Repository Structure

```text
Infosys-PlantBot/
├── app.py                         # Streamlit web application & tabbed UI
├── utils.py                       # Multimodal inference engine & weighted fusion logic
├── data/                          # Domain knowledge base & label sets
│   ├── class_names.txt            # 38 PlantVillage disease class names
│   └── disease_info.json          # Curated causes, symptoms & treatment protocols
├── models/                        # Local weights directory (Ignored by Git)
│   ├── plant_doctor_model.pth     # ResNet34 PyTorch state dict
│   └── plant_disease_bert/        # Fine-tuned DistilBERT transformer checkpoint
├── training/                      # Colab retraining scripts & runbooks
│   ├── train_visual_model.py      # ResNet34 training pipeline
│   ├── train_text_model.py        # DistilBERT fine-tuning pipeline
│   └── update_symptoms.py         # Knowledge base symptom enrichment script
├── docs/                          # Project specifications & certifications
│   ├── AI_PlantDocBot_Specification.pdf # Infosys milestone project requirements
│   └── certificates/              # Infosys Springboard completion credentials
│       ├── InfosysSpringboard.pdf
│       └── InfosysSpringboard.jpg
├── AGENTS.md                      # Living workspace governance contract
├── requirements.txt               # Python package dependencies
├── LICENSE                        # MIT License
├── README.md                      # Production portfolio documentation
└── .gitignore                     # Git exclusions (models, weights, caches)
```

---

## Certification & Attribution

This project was developed and evaluated as part of the **Infosys Springboard AI Project Internship** program.
- **Specification Document**: [`docs/AI_PlantDocBot_Specification.pdf`](docs/AI_PlantDocBot_Specification.pdf)
- **Certificates**: [`docs/certificates/`](docs/certificates/)

---

## License

This project is licensed under the [MIT License](LICENSE).
