# Libs
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from datasets import load_dataset


# Configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_wav2vec = "facebook/wav2vec2-base-960h"


# Wav2Vec2.0 Feature Extractor

class Wav2VecFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.model.eval().to(device)

    def forward(self, raw_waveform, sampling_rate):
        # raw_waveform: tensor of shape [1, T]
        input_values = self.processor(raw_waveform.squeeze(0).numpy(), sampling_rate=sampling_rate, return_tensors="pt", padding=True).input_values
        input_values = input_values.to(device)  # shape: [1, T]

        with torch.no_grad():
            outputs = self.model(input_values)
            hidden_states = outputs.last_hidden_state  # shape: [1, T', 768]

        return hidden_states
        

# Acoustic Encoder: Conv1D (kernel size = 5) + LSTM (hidden layer = 1024 nodes)
class AcousticEncoder(nn.Module):
    def __init__(self, input_dim=768, conv_out_dim=1024, lstm_hidden=1024):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=conv_out_dim, kernel_size=5, padding=2)
        self.lstm = nn.LSTM(input_size=conv_out_dim, hidden_size=lstm_hidden, batch_first=True)

    def forward(self, features):
        # features: [B, T, D]
        features = features.transpose(1, 2)  # [B, D, T] for Conv1D
        conv_out = self.conv1d(features)     # [B, C, T]
        conv_out = conv_out.transpose(1, 2)  # [B, T, C] for LSTM
        lstm_out, _ = self.lstm(conv_out)   # [B, T, H]
        return lstm_out


# Full acoustic branch
class AcousticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = Wav2VecFeatureExtractor()
        self.encoder = AcousticEncoder()

    def forward(self, raw_waveform, sampling_rate):
        with torch.no_grad():  # Freezing wav2vec for now
            features = self.feature_extractor(raw_waveform, sampling_rate)
        task_features = self.encoder(features)
        return task_features  # [B, T, 256]



# # Inference on LibriSpeech sample

# model = AcousticModel().to(device)
# model.eval()

# waveform = torch.tensor(audio_array, dtype=torch.float32)
# print("Wav2vec2 waveform shape", waveform.shape)

# with torch.no_grad():
#     # task_features = model([waveform], sampling_rate)
#     task_features = model(waveform.unsqueeze(0), sampling_rate)

# print("Task-specific acoustic features shape:", task_features.shape)