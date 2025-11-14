# Test Script for CNN + BiLSTM
# Evaluates trained model on test sequences
# Author: Naman Goenka
# Date: Dec 25th 2020

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from cnn_bilstm_model import CNNBiLSTM
from train_cnn_bilstm import GitaSequenceDataset


def test_model(model, test_loader, device):
    """Test the model and calculate metrics."""
    model.eval()
    
    total_correct = 0
    total_chars = 0
    word_correct = 0
    total_words = 0
    
    print("Testing model...")
    
    with torch.no_grad():
        for images, labels, seq_lengths in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            seq_lengths = torch.tensor(seq_lengths)
            
            # Get predictions
            outputs = model(images, seq_lengths)
            predictions = torch.argmax(outputs, dim=-1)
            
            # Character-level accuracy
            mask = labels != -1
            correct = ((predictions == labels) & mask).sum().item()
            total = mask.sum().item()
            
            total_correct += correct
            total_chars += total
            
            # Word-level accuracy (all characters must be correct)
            for i in range(len(labels)):
                seq_mask = mask[i]
                if seq_mask.sum() == 0:
                    continue
                
                pred_seq = predictions[i][seq_mask]
                label_seq = labels[i][seq_mask]
                
                if torch.all(pred_seq == label_seq):
                    word_correct += 1
                total_words += 1
    
    char_accuracy = total_correct / total_chars if total_chars > 0 else 0
    word_accuracy = word_correct / total_words if total_words > 0 else 0
    
    return char_accuracy, word_accuracy


def main():
    # Configuration
    NUM_CLASSES = 102
    CNN_FEATURE_DIM = 128
    LSTM_HIDDEN_DIM = 128
    NUM_LSTM_LAYERS = 1
    BATCH_SIZE = 8
    MAX_SEQ_LENGTH = 15
    MODEL_PATH = 'best_cnn_bilstm_model.pth'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading test dataset...")
    dataset = GitaSequenceDataset(
        data_dir='../training_data_augmented',
        max_seq_length=MAX_SEQ_LENGTH
    )
    
    # Use 20% as test set
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Test sequences: {test_size}")
    
    # Load model
    print("Loading model...")
    model = CNNBiLSTM(
        num_classes=NUM_CLASSES,
        cnn_feature_dim=CNN_FEATURE_DIM,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        num_lstm_layers=NUM_LSTM_LAYERS
    ).to(device)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Validation accuracy: {checkpoint['val_acc']:.4f}")
    
    # Test
    char_acc, word_acc = test_model(model, test_loader, device)
    
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Character-level Accuracy: {char_acc:.4f} ({char_acc*100:.2f}%)")
    print(f"Word-level Accuracy: {word_acc:.4f} ({word_acc*100:.2f}%)")
    print("="*50)


if __name__ == '__main__':
    main()
