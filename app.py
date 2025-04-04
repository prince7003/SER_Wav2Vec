import streamlit as st
import soundfile as sf
import os
import torch
import torch.nn.functional as F
import json
import librosa
import numpy as np
from transformers import Wav2Vec2Processor
from model.classifier import EmotionClassifier

# Load processor locally
processor = Wav2Vec2Processor.from_pretrained("./saved_processor", local_files_only=True)

# Load model config
with open("./saved_model/model_config.json", "r") as f:
    model_config = json.load(f)

# Load emotion mappings
with open("emotion_mappings.json", "r") as f:
    mappings = json.load(f)
    id2emotion = mappings["id2emotion"]

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionClassifier(
    model_config["model_name"], 
    model_config["num_labels"],
    local_files_only=True
)
model.load_state_dict(torch.load("./saved_model/best_model.pt", map_location=device))
model.to(device)
model.eval()

st.title("Audio Emotion Recognition")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display audio player
    st.audio("temp_audio.wav")

    if st.button("Predict Emotion"):
        # Preprocess audio
        audio, sr = librosa.load("temp_audio.wav", sr=16000)
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Convert to tensor
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        
        # Get prediction
        with torch.no_grad():
            logits = model(input_values)
            probs = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
        
        # Get emotion and confidence
        emotion = id2emotion[str(predicted_class)]
        confidence = probs[0][predicted_class].item()
        
        # Get all emotion probabilities
        emotion_probs = {id2emotion[str(i)]: probs[0][i].item() for i in range(len(id2emotion))}
        
        # Display results
        st.write(f"Predicted Emotion: {emotion}")
        st.write(f"Confidence: {confidence:.2f}")
        
        # Display all probabilities
        st.write("Emotion Probabilities:")
        for emotion, prob in emotion_probs.items():
            st.write(f"{emotion}: {prob:.2f}")
        
        # Create a bar chart of probabilities
        st.bar_chart(emotion_probs)

    # Remove temporary file
    if os.path.exists("temp_audio.wav"):
        os.remove("temp_audio.wav")