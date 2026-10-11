import streamlit as st
from PIL import Image
import torch
import numpy as np
from utils import load_models_and_data, predict_image, predict_text_bert, predict_text_sbert

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
    .main { background-color: #f5f7f9; }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover { background-color: #1b5e20; }
    .title-text {
        color: #1b5e20;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .subtitle-text { text-align: center; color: #555; }
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
# LOAD RESOURCES
# ==========================================
model_viz, model_text, bert_pipe, class_names, disease_info, kb_embeddings, status = load_models_and_data()

# ==========================================
# UI LAYOUT
# ==========================================
st.markdown("<h1 class='title-text'>🌿 PlantDocBot</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Multi-Modal Diagnosis: Image, Text, or Hybrid Fusion</p>", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("System Status")
    if "SUCCESS" in str(status):
        st.success("✅ Models Loaded")
        text_model_name = "Custom BERT (Expert)" if bert_pipe else "SBERT (Semantic Search)"
        st.caption(f"• Visual: ResNet34\n• Text: {text_model_name}")
    else:
        st.error(f"❌ Error: {status}")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📷 Image Only", "💬 Text Only", "🧬 Hybrid (Best)"])

# 1. IMAGE ONLY
with tab1:
    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"], key="img_only")
    if uploaded_file and st.button("Analyze Image 📷"):
        col1, col2 = st.columns([1, 2])
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, width="stretch")
        with col2:
            if model_viz:
                pred_class, score, _ = predict_image(model_viz, image, class_names)

                st.markdown(f"### Result: **{pred_class}**")
                st.progress(int(score))
                st.caption(f"Confidence: {score:.1f}%")

                info = disease_info.get(pred_class, {})
                if info:
                    st.info(f"**Description:** {info.get('description')}")
                    st.success(f"**Treatment:** {info.get('treatment')}")
            else:
                st.error("Visual model not loaded.")

# 2. TEXT ONLY
with tab2:
    user_text = st.text_area("Describe the symptoms", placeholder="e.g. Tomato leaves have dark brown circles...")
    if user_text and st.button("Analyze Symptoms 💬"):
        if bert_pipe:
            best_class, best_score, _ = predict_text_bert(bert_pipe, user_text, class_names)
            st.caption("Used Custom BERT Model")
        else:
            best_class, best_score, _, _ = predict_text_sbert(model_text, kb_embeddings, user_text, class_names)
            st.caption("Used SBERT Semantic Search")

        st.markdown(f"### Result: **{best_class}**")
        st.progress(int(min(best_score, 100)))
        st.caption(f"Confidence: {best_score:.1f}%")

        info = disease_info.get(best_class, {})
        if info:
            st.info(f"**Description:** {info.get('description')}")
            st.success(f"**Treatment:** {info.get('treatment')}")

# 3. HYBRID
with tab3:
    st.info("💡 Combines **Image (70%)** and **Text (30%)**.")
    h_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"], key="hybrid_img")
    h_text = st.text_area("Describe Symptoms (Optional)", key="hybrid_txt")

    if h_file and st.button("Run Hybrid Analysis 🧬"):
        # Image
        image = Image.open(h_file).convert('RGB')
        st.image(image, width="stretch")
        _, _, img_probs = predict_image(model_viz, image, class_names)

        # Text
        text_scores = np.zeros(len(class_names))
        if h_text.strip():
            if bert_pipe:
                _, _, b_idx = predict_text_bert(bert_pipe, h_text, class_names)
                pred = bert_pipe(h_text)[0]
                text_scores[b_idx] = pred['score']
                st.caption("Mixed with BERT Prediction")
            else:
                _, _, cos_scores, _ = predict_text_sbert(model_text, kb_embeddings, h_text, class_names)
                text_scores = np.maximum(cos_scores, 0)
        else:
            st.caption("ℹ️ No text provided. Using 100% Image Mode.")

        # Weighted Combination
        if h_text.strip():
            final_scores = (img_probs * 0.7) + (text_scores * 0.3)
        else:
            final_scores = img_probs

        best_idx = np.argmax(final_scores)
        best_class = class_names[best_idx]
        final_conf = final_scores[best_idx] * 100

        st.markdown(f"""
        <div class='prediction-card'>
            <h2>Hybrid Diagnosis: <span style='color: #2e7d32;'>{best_class}</span></h2>
            <p>Combined Confidence: <b>{final_conf:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

        info = disease_info.get(best_class, {})
        if info:
            st.info(f"**Description:** {info.get('description')}")
            st.success(f"**Treatment:** {info.get('treatment')}")
