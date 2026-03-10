print("Training started...")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from dataset_loader import RAVDESSDataset
from models.multimodal_model import MultimodalModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def train():

    dataset = RAVDESSDataset(config.METADATA_FILE)

    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    model = MultimodalModel().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(config.EPOCHS):

        total_loss = 0

        for text, audio, visual, label in loader:
            
            print("Text shape:", text.shape)
            print("Audio shape:", audio.shape)
            print("Visual shape:", visual.shape)
            print("Batch running...")

            text = text.to(device)
            audio = audio.to(device)
            visual = visual.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()

            output = model(text, audio, visual)

            loss = criterion(output, label)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{config.EPOCHS}, Loss: {total_loss:.4f}")

    import os

    os.makedirs("../models", exist_ok=True)
    print("Saving model to:", os.path.abspath("../models/multimodal_model.pth"))

    torch.save(model.state_dict(), "../models/multimodal_model.pth")

    print("Model saved successfully")
   
if __name__ == "__main__":
    train()