# Training Script for CNN + BiLSTM
# Trains on Gita dataset augmentation output
# Author: Naman Goenka
# Date: Dec 25th 2020

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

from cnn_bilstm_model import CNNBiLSTM


class GitaSequenceDataset(Dataset):
    """using Gita augmented data."""
    
    def __init__(self, data_dir='../training_data_augmented', max_seq_length=15):
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        
        self.sequences = self._parse_dataset()
        
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _parse_dataset(self):
        """Parse training_data_augmented directory structure."""
        from pathlib import Path
        
        sequences = {}
        
        for class_dir in Path(self.data_dir).iterdir():
            if not class_dir.is_dir():
                continue
            
            # Example: class_355_bhi
            class_name = class_dir.name
            parts = class_name.split('_')
            if len(parts) < 2:
                continue
            class_idx = int(parts[1])
            
            # Process images
            for img_path in class_dir.glob('*.png'):
                # Parse filename: verse_wX_lY.png
                stem = img_path.stem
                parts = stem.split('_')
                
                if len(parts) >= 3:
                    verse_id = parts[0]
                    word_idx = int(parts[1][1:]) if parts[1].startswith('w') else 0
                    char_idx = int(parts[2][1:]) if parts[2].startswith('l') else 0
                    
                    word_id = f"{verse_id}_w{word_idx}"
                    
                    if word_id not in sequences:
                        sequences[word_id] = []
                    
                    sequences[word_id].append({
                        'char_idx': char_idx,
                        'image_path': str(img_path),
                        'class_label': class_idx
                    })
        
        # Sort characters in each sequence
        for word_id in sequences:
            sequences[word_id] = sorted(sequences[word_id], key=lambda x: x['char_idx'])
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        word_id = list(self.sequences.keys())[idx]
        chars = self.sequences[word_id]
        
        # Load images and labels
        images = []
        labels = []
        
        for char in chars:
            img = Image.open(char['image_path']).convert('RGB')
            img = self.transform(img)
            images.append(img)
            labels.append(char['class_label'])
        
        # Pad or truncate
        seq_len = len(images)
        if seq_len > self.max_seq_length:
            images = images[:self.max_seq_length]
            labels = labels[:self.max_seq_length]
            seq_len = self.max_seq_length
        elif seq_len < self.max_seq_length:
            pad_size = self.max_seq_length - seq_len
            images.extend([torch.zeros(3, 32, 32)] * pad_size)
            labels.extend([-1] * pad_size)
        
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return images, labels, seq_len


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels, seq_lengths in tqdm(train_loader, desc='Training'):
        images = images.to(device)
        labels = labels.to(device)
        seq_lengths = torch.tensor(seq_lengths)
        
        optimizer.zero_grad()
        outputs = model(images, seq_lengths)
        
        # Calculate loss (ignore padding with label=-1)
        outputs_flat = outputs.view(-1, outputs.size(-1))
        labels_flat = labels.view(-1)
        loss = criterion(outputs_flat, labels_flat)
        
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        mask = labels != -1
        pred = torch.argmax(outputs, dim=-1)
        correct += ((pred == labels) & mask).sum().item()
        total += mask.sum().item()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, seq_lengths in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            labels = labels.to(device)
            seq_lengths = torch.tensor(seq_lengths)
            
            outputs = model(images, seq_lengths)
            
            outputs_flat = outputs.view(-1, outputs.size(-1))
            labels_flat = labels.view(-1)
            loss = criterion(outputs_flat, labels_flat)
            
            mask = labels != -1
            pred = torch.argmax(outputs, dim=-1)
            correct += ((pred == labels) & mask).sum().item()
            total += mask.sum().item()
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def main():
    # Configuration
    NUM_CLASSES = 102
    CNN_FEATURE_DIM = 128
    LSTM_HIDDEN_DIM = 128
    NUM_LSTM_LAYERS = 1
    BATCH_SIZE = 8
    EPOCHS = 30
    LEARNING_RATE = 0.001
    MAX_SEQ_LENGTH = 15
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading dataset...")
    dataset = GitaSequenceDataset(
        data_dir='../training_data_augmented',
        max_seq_length=MAX_SEQ_LENGTH
    )
    
    print(f"Total sequences: {len(dataset)}")
    
    # Train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train sequences: {train_size}, Val sequences: {val_size}")
    
    # Model
    print("Creating model...")
    model = CNNBiLSTM(
        num_classes=NUM_CLASSES,
        cnn_feature_dim=CNN_FEATURE_DIM,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        num_lstm_layers=NUM_LSTM_LAYERS
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_val_acc = 0
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_cnn_bilstm_model.pth')
            print(f"Saved best model (val_acc: {val_acc:.4f})")
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()
