# AGENTS.md - Repository Architecture & Development Guidelines

This repository implements **PlantDocBot**, a multimodal AI diagnostic platform developed during the **Infosys Springboard** program. It integrates Computer Vision (ResNet34) and Natural Language Processing (Fine-tuned DistilBERT / Sentence-Transformers) with a Streamlit interface.

All agents modifying this codebase must adhere strictly to these architectural boundaries and guidelines:

---

## 1. Directory Structure & Component Boundaries

- **Web Application Layer (`app.py`)**:
  - Contains the Streamlit user interface, tabs (Image Only, Text Only, Hybrid Mode), and presentation cards.
  - Must remain decoupled from heavy model initialization logic by delegating to `utils.py`.

- **Inference & Multimodal Fusion Engine (`utils.py`)**:
  - `load_models_and_data()`: Resource loader cached with `@st.cache_resource`.
  - `predict_image()`: Visual inference pipeline running leaf images through ResNet34 with standard ImageNet normalization.
  - `predict_text_bert()` / `predict_text_sbert()`: Natural language symptom diagnostic using fine-tuned DistilBERT (or SBERT semantic similarity fallback).
  - Hybrid fusion: Computes weighted probability blending ($0.7 \times P_{\text{visual}} + 0.3 \times P_{\text{text}}$).

- **Data & Knowledge Base (`data/`)**:
  - `class_names.txt`: The canonical list of 38 plant disease classes from the PlantVillage dataset.
  - `disease_info.json`: Curated domain knowledge mapping all 38 classes to symptoms, causes, descriptions, and treatment protocols.

- **Model Training Scripts (`training/`)**:
  - `train_visual_model.py`: Standalone Colab training script for fine-tuning ResNet34 on the PlantVillage dataset.
  - `train_text_model.py`: Standalone Colab training script for fine-tuning DistilBERT on text-image symptom pairs.
  - `update_symptoms.py`: Utility script to enrich and format symptom knowledge.

- **Documentation & Certification (`docs/`)**:
  - `AI_PlantDocBot_Specification.pdf`: Formal project specification and milestone evaluation criteria.
  - `certificates/`: Official Infosys Springboard internship and project completion credentials (`InfosysSpringboard.pdf`, `InfosysSpringboard.jpg`).

---

## 2. Storage & Git Hygiene Constraints (CRITICAL)

- **Zero Weights in Git**: Model weight files (`plant_doctor_model.pth`, `model.safetensors`, `training_args.bin`, etc.) exceed hundreds of megabytes and must **never** be committed to Git.
- **Enforce `.gitignore`**: Keep `models/`, `*.pth`, `*.safetensors`, `*.bin`, `*.onnx`, and `.venv/` strictly ignored.
- **External Weights**: Weights are distributed via Google Drive or re-generated via the scripts in `training/`.

---

## 3. Encoding & UI Quality Invariants

- All files must be saved with UTF-8 encoding without Byte Order Marks (BOM).
- Ensure Streamlit emoji symbols (`🌿`, `📷`, `💬`, `🧬`, `✅`, `💡`, `❌`) remain clean and uncorrupted across Windows and Unix platforms.
