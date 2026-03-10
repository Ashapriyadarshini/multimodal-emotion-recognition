import os
import cv2
import torch
import librosa
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from transformers import BertTokenizer


class RAVDESSDataset(Dataset):

    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.data)

    # -------- VISUAL FEATURE --------
    def extract_visual(self, video_path):

        try:
            cap = cv2.VideoCapture(video_path)

            ret, frame = cap.read()

            if not ret:
                raise Exception("Frame not found")

            frame = cv2.resize(frame, (224,224))

            frame = frame.astype(np.float32) / 255.0

            frame = np.transpose(frame,(2,0,1))

            return torch.tensor(frame)

        except:

            # fallback if corrupted video
            return torch.zeros((3,224,224))

    # -------- AUDIO FEATURE --------
    def extract_audio(self, video_path):

        try:

            y, sr = librosa.load(video_path, sr=16000)

            mel = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=2048,
                hop_length=512,
                n_mels=128
            )

            mel = librosa.power_to_db(mel)

            mel = (mel - np.mean(mel)) / (np.std(mel)+1e-6)

            mel = torch.tensor(mel).unsqueeze(0)

            return mel

        except:

            return torch.zeros((1,128,128))

    # -------- TEXT FEATURE --------
    def extract_text(self, text):

        tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        return tokens["input_ids"].squeeze(0)

    # -------- GET ITEM --------
    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        video_path = row["filepath"]

        # RAVDESS has no transcript → dummy text
        text = "emotion speech sample"

        emotion = row["emotion"]

        emotion_map = {
            "neutral":0,
            "calm":1,
            "happy":2,
            "sad":3,
            "angry":4,
            "fearful":5,
            "disgust":6,
            "surprised":7
        }

        label = emotion_map[emotion]

        label = torch.tensor(label)

        # return dummy tensors temporarily for testing
        text_tensor = torch.randint(0, 30522, (512,))
        audio_tensor = torch.randn(1, 64, 64)
        visual_tensor = torch.randn(3,224,224)

        return text_tensor, audio_tensor, visual_tensor, label