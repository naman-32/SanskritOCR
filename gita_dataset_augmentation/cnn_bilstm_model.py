# CNN + BiLSTM for sequence-based character recognition
# Reuses existing CNN architecture from classification/letter_level/cnn.py
# Author: Naman Goenka
# Date: Dec 25th 2020

import torch
import torch.nn as nn
import torch.nn.functional as F
from keras.models import load_model
import numpy as np


class CNNBiLSTM(nn.Module):
    
    def __init__(self, num_classes=102, cnn_feature_dim=128, 
                 lstm_hidden_dim=128, num_lstm_layers=1):
        """
        Args:
            num_classes: Number of character classes (102 for Sanskrit)
            cnn_feature_dim: Dimension of CNN output features
            lstm_hidden_dim: Hidden dimension of BiLSTM
            num_lstm_layers: Number of BiLSTM layers
        """
        super(SimpleCNNBiLSTM, self).__init__()
        
        self.num_classes = num_classes
        self.cnn_feature_dim = cnn_feature_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        
        # 32x32x3
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  
        # 32x32 -> 16x16
        
        self.conv4 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  
        # 16x16 -> 8x8
        
        self.conv7 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  
        # 8x8 -> 4x4
        
        self.fc_cnn = nn.Linear(256, cnn_feature_dim)
        self.dropout_cnn = nn.Dropout(0.3)
        
        self.bilstm = nn.LSTM(
            input_size=cnn_feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # (bidirectional so *2)
        self.fc_out = nn.Linear(lstm_hidden_dim * 2, num_classes)
    
    def extract_cnn_features(self, x):
        """
        Extract features from a single character image using CNN.
        x: (batch, 3, 32, 32)
        Returns features: (batch, cnn_feature_dim)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout_cnn(F.relu(self.fc_cnn(x)))
        
        return x
    
    def forward(self, x, seq_lengths=None):
        """
        Forward pass through CNN + BiLSTM.
        x: (batch_size, seq_length, 3, 32, 32)
        seq_lengths: (batch_size,) actual sequence lengths
        output: (batch_size, seq_length, num_classes)
        """
        batch_size, seq_length = x.size(0), x.size(1)
        
        # Reshape to process all characters at once
        x = x.view(batch_size * seq_length, 3, 32, 32)
        cnn_features = self.extract_cnn_features(x)  
        # (batch*seq_len, cnn_feature_dim)
        
        # Reshape
        cnn_features = cnn_features.view(batch_size, seq_length, -1)
        
        # # Pack padded sequences for efficiency
        if seq_lengths is not None:
            cnn_features = nn.utils.rnn.pack_padded_sequence(
                cnn_features, seq_lengths.cpu(), 
                batch_first=True, enforce_sorted=False
            )
        
        lstm_out, _ = self.bilstm(cnn_features)
        
        # Unpack if packed
        if seq_lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        
        output = self.fc_out(lstm_out)  
        # (batch, seq_len, num_classes)
        
        return output
    
    def predict(self, x, seq_lengths=None):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, seq_lengths)
            predictions = torch.argmax(logits, dim=-1)
        return predictions


def load_pretrained_cnn_weights(model, keras_model_path):
    try:
        from keras.models import load_model as keras_load
        keras_model = keras_load(keras_model_path)
        print(f"Loaded Keras model from {keras_model_path}")
        return keras_model
    except Exception as e:
        print(f"Could not load Keras model: {e}")
        return None


if __name__ == "__main__":
    print("Testing CNN + BiLSTM Model")
    
    model = CNNBiLSTM(
        num_classes=102,
        cnn_feature_dim=128,
        lstm_hidden_dim=128,
        num_lstm_layers=1
    )
    
    x = torch.randn(2, 10, 3, 32, 32)
    seq_lengths = torch.tensor([10, 8])
    
    output = model(x, seq_lengths)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (batch=2, seq_len=10, num_classes=102)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n CNN +BiLSTM Model created")
