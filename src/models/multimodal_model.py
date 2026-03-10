import torch
import torch.nn as nn

from models.visual_model import VisualModel
from models.audio_model import AudioModel
from models.text_model import TextModel
from fusion.attention_fusion import AttentionFusion


class MultimodalModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.visual = VisualModel()

        self.audio = AudioModel()

        self.text = TextModel()

        self.fusion = AttentionFusion()

        self.classifier = nn.Sequential(

            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,8)

        )

    def forward(self,text,audio,visual):

        v = self.visual(visual)

        a = self.audio(audio)

        t = self.text(text)

        fused = self.fusion(v,a,t)

        out = self.classifier(fused)

        return out