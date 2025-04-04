import os
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import json
from dotenv import load_dotenv
from config import config
import os

load_dotenv()

HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

def setup():
    # Create directories if they don't exist
    os.makedirs("./saved_processor", exist_ok=True)
    os.makedirs("./saved_model", exist_ok=True)
    
    print("Downloading and saving processor...")
    # Download and save processor
    processor = Wav2Vec2Processor.from_pretrained(
        config.model_name,
        token = HF_TOKEN
    )
    processor.save_pretrained("./saved_processor")
    print("Processor saved")

    
    print("Downloading and saving base model...")
    # Download and save base wav2vec model configuration
    base_model = Wav2Vec2Model.from_pretrained(
        config.model_name,
        token = HF_TOKEN
    )
    base_model.save_pretrained("./saved_base_model")
    
    # Save model configuration
    model_config = {
        "model_name": "facebook/wav2vec2-large-960h-lv60-self",
        "num_labels": 8
    }
    with open("./saved_model/model_config.json", "w") as f:
        json.dump(model_config, f)
    
    print("Setup complete! You can now run the app without Hugging Face dependency.")

if __name__ == "__main__":
    setup()


 


