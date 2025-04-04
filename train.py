import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Processor
from config import config
from data.dataset import prepare_ravdess_dataset
from model.classifier import EmotionClassifier
from utils.training_utils import compute_metrics

from dotenv import load_dotenv
import os
load_dotenv()
HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

def train_model(train_dataset, eval_dataset, model, config):
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_f1 = 0.0
    best_model_state = None

    print(f"Training on device: {device}")

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for batch in progress_bar:
            input_values = batch["input_values"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(input_values)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                input_values = batch["input_values"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_values)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = compute_metrics(np.array(all_preds), np.array(all_labels))

        print(f"Epoch {epoch + 1}/{config.num_epochs}, Train Loss: {avg_train_loss:.4f}, "
              f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['weighted_f1']:.4f}")

        if metrics['weighted_f1'] > best_f1:
            best_f1 = metrics['weighted_f1']
            best_model_state = model.state_dict().copy()

    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), "./saved_model/best_model.pt")

    return model, metrics

def main():
    processor = Wav2Vec2Processor.from_pretrained("./saved_processor", local_files_only=True)

    print("Preparing RAVDESS dataset...")
    train_dataset, test_dataset, emotion2id, id2emotion = prepare_ravdess_dataset(processor)

    config.num_labels = len(emotion2id)

    print("Initializing model...")
    model = EmotionClassifier(config.model_name, config.num_labels)

    print("Training model...")
    model, metrics = train_model(train_dataset, test_dataset, model, config)

    print(f"Final metrics: Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['weighted_f1']:.4f}")
    print("Training and evaluation complete!")

if __name__ == "__main__":
    main()
