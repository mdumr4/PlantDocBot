import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

# ==========================================
# CONFIGURATION & STYLE
# ==========================================
st.set_page_config(
    page_title="PlantDocBot",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for a Premium Look
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
    }
    .stAlert {
        border-radius: 10px;
    }
    .title-text {
        color: #1b5e20;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .subtitle-text {
        text-align: center;
        color: #555;
    }
    .prediction-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# MODEL LOADING
# ==========================================
@st.cache_resource
def load_resources():
    """
    Loads the model, class names, and disease info.
    Cached prevents reloading on every button click.
    """
    # 1. Load Class Names
    if not os.path.exists("class_names.txt"):
        return None, None, None, "MISSING_CLASSES"

    with open("class_names.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # 2. Load Disease Info (Treatments)
    disease_info = {}
    if os.path.exists("disease_info.json"):
        with open("disease_info.json", "r") as f:
            disease_info = json.load(f)

    # 3. Load Model
    if not os.path.exists("plant_doctor_model.pth"):
        return None, class_names, disease_info, "MISSING_MODEL"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Recreate the exact architecture used in training (ResNet34)
    model = models.resnet34(weights=None) # No need for internet weights, we have our own
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    try:
        model.load_state_dict(torch.load("plant_doctor_model.pth", map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        return None, class_names, disease_info, f"MODEL_ERROR: {str(e)}"

    return model, class_names, disease_info, "SUCCESS"

model, class_names, disease_info, status = load_resources()

# ==========================================
# PREPROCESSING
# ==========================================
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0) # Add batch dimension

# ==========================================
# UI LAYOUT
# ==========================================
st.markdown("<h1 class='title-text'>🌿 PlantDocBot</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>AI-Powered Plant Disease Diagnosis & Treatment</p>", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("Project Status")

    if status == "SUCCESS":
        st.success("✅ System Online")
        st.info(f"Loaded {len(class_names)} Disease Classes")
    elif status == "MISSING_MODEL":
        st.error("❌ Model Missing")
        st.warning("Please place 'plant_doctor_model.pth' in this folder.")
    elif status == "MISSING_CLASSES":
        st.error("❌ Classes Missing")
        st.warning("Please place 'class_names.txt' in this folder.")
    else:
        st.error(f"System Error: {status}")

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("1. Upload a clear photo of a plant leaf.")
    st.markdown("2. The AI will analyze the patterns.")
    st.markdown("3. Get instant diagnosis and treatment advice.")

# --- MAIN CONTENT ---

# File Uploader
uploaded_file = st.file_uploader("Upload a Plant Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Leaf', use_container_width=True)

    # Prediction Button
    if st.button("Analyze Plant 🔍"):
        if status != "SUCCESS":
            st.error("Cannot predict: System files are missing (Model or Class Names).")
        else:
            with st.spinner(" analyzing leaf texture and patterns..."):
                # Prediction Logic
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                img_tensor = process_image(image).to(device)

                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)

                    predicted_class = class_names[predicted_idx.item()]
                    confidence_score = confidence.item() * 100

            # Display Results
            st.markdown(f"""
            <div class='prediction-card'>
                <h2>Diagnosis: <span style='color: #2e7d32;'>{predicted_class}</span></h2>
                <p>Confidence: <b>{confidence_score:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

            # Recommendation Section
            st.markdown("### 💊 Treatment & Recommendations")

            # Try to find infoKey that matches the predicted class
            # The disease info keys might not match exactly if loaded from a different format,
            # so we try direct lookup or simple normalization.
            info = disease_info.get(predicted_class)

            if info:
                st.info(f"**Description:** {info.get('description', 'No description.')}")
                st.success(f"**Treatment:** {info.get('treatment', 'No specific treatment in database.')}")
            else:
                # Fallback if specific json key not found
                st.warning("No specific treatment details found in database for this exact label.")
                st.markdown(f"**Suggested Actions:** Search online for *'{predicted_class} treatment'*.")

else:
    # Empty State
    st.info("👆 Upload an image to get started.")
