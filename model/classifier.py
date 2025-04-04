import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
    

class EmotionClassifier(nn.Module):
    def __init__(self, model_name, num_labels, local_files_only=False):
        super().__init__()
        if local_files_only:
            self.wav2vec = Wav2Vec2Model.from_pretrained(
                "./saved_base_model",
                local_files_only=True
            )
        else:
            self.wav2vec = Wav2Vec2Model.from_pretrained(
                model_name,
                token= HF_TOKEN
            )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(1024, num_labels)

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            pooled_output = torch.sum(hidden_states * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        else:
            pooled_output = torch.mean(hidden_states, dim=1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
