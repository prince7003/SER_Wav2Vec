import torch
import torch.nn.functional as F
import json
from transformers import Wav2Vec2Processor
from model.classifier import EmotionClassifier
from utils.audio_utils import preprocess_audio
from config import config

class EmotionRecognitionPipeline:
    def __init__(self, model_path="./saved_model/best_model.pt", processor_path="./saved_processor"):
        # Load emotion mappings
        with open("emotion_mappings.json", "r") as f:
            mappings = json.load(f)
            self.id2emotion = mappings["id2emotion"]
            self.emotion2id = mappings["emotion2id"]

        # Load model config
        with open("./saved_model/model_config.json", "r") as f:
            model_config = json.load(f)

        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EmotionClassifier(
            model_config["model_name"], 
            model_config["num_labels"],
            local_files_only=True
        )
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        # Load processor locally
        self.processor = Wav2Vec2Processor.from_pretrained(processor_path, local_files_only=True)
        self.device = device
        self.sampling_rate = 16000
