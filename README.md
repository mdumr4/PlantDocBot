# 🌿 PlantDocBot - Multi-Modal Plant Disease Diagnosis

PlantDocBot is an advanced AI assistant that diagnoses plant diseases using:
1.  **Image Analysis**: ResNet34 CNN (95%+ Accuracy)
2.  **Text Analysis**: Fine-tuned DistilBERT (Context-aware, handles negations)
3.  **Hybrid Mode**: Weighted fusion of Visual (70%) and Textual (30%) features.

## 🚀 Setup & Installation

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Model Setup**
    Since model files are large, they are **not included** in this repository. You have two options:

    *   **Option A: Train them yourself (Recommended)**
        *   Run `training/train_visual_model.py` in Colab to generate `plant_doctor_model.pth`.
        *   Run `training/train_text_model.py` in Colab to generate the `plant_disease_bert` folder.
        *   Place them in the `models/` directory.

    *   **Option B: Download Pre-trained Weights**
        *   [Link to Google Drive](https://drive.google.com/drive/folders/1TBWM30fXpK_mGARJQryr7iU2HtItPa1X?usp=sharing)
        *   Download and extract into `models/`.

3.  **Run the App**
    ```bash
    streamlit run app.py
    ```

## 🧠 Model Training

The models were trained using Google Colab.

*   **Visual Model**: `Model/colab_training.py` (ResNet34 on PlantVillage)
*   **Text Model**: `Model/train_text_model_colab.py` (DistilBERT on Text-Image Pairs)

## 📂 Project Structure

*   `app.py`: Main UI Application
*   `utils.py`: Logic Module
*   `models/`: Hosts `plant_doctor_model.pth` and `plant_disease_bert/` folder.
*   `data/`: Hosts `class_names.txt` and `disease_info.json`.
*   `training/`: Contains all training scripts (formerly `Model/`).
