# Allow imports from parent directory
import os,sys
dir_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(dir_path)
sys.path.append(parent_dir)

from util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(2025)

# Data
BATCH_SIZE = 2048
train_dataloader, val_dataloader, test_dataloader, voc = load_data(batch_size=BATCH_SIZE) 

num_class = 5
vocab_size = len(voc)

# BoW Model
EMBED_DIM_BOW = 64
bow_model = BoW(vocab_size, EMBED_DIM_BOW, num_class).to(device)
bow_model.load_state_dict(torch.load("../Models/bow_model_SGD_3E.pth", weights_only=True))
print("BoW Model loaded")

# RNN Model
EMBED_DIM_RNN = 20
RNN_HIDDEN_DIM = 128
FC_HIDDEN_DIM = 128
rnn_model = RNN(vocab_size, EMBED_DIM_RNN, RNN_HIDDEN_DIM, FC_HIDDEN_DIM, num_classes=num_class).to(device)
rnn_model.load_state_dict(torch.load("../Models/rnn_model_3E.pth", weights_only=True))
print("RNN Model loaded")

# LSTM Model
EMBED_DIM_LSTM = 20
LSTM_HIDDEN_DIM = 128
FC_HIDDEN_DIM = 128
lstm_model = LSTM(vocab_size, EMBED_DIM_LSTM, LSTM_HIDDEN_DIM, FC_HIDDEN_DIM, num_classes=num_class).to(device)
lstm_model.load_state_dict(torch.load("../Models/LSTM_model_3E.pth", weights_only=True))
print("LSTM Model loaded")

# Plot length accuracy
bins_bow, acc_bow = compute_length_accuracy(bow_model, test_dataloader, device)
print("Found BoW length accuracy")
bins_rnn, acc_rnn = compute_length_accuracy(rnn_model, test_dataloader, device)
print("Found RNN length accuracy")
bins_lstm, acc_lstm = compute_length_accuracy(lstm_model, test_dataloader, device)
print("Found LSTM length accuracy")

import matplotlib.pyplot as plt
plt.plot(bins_bow, acc_bow, label="BoW Model")
plt.plot(bins_rnn, acc_rnn, label="RNN Model")
plt.plot(bins_lstm, acc_lstm, label="LSTM Model")
plt.xlabel("Sequence Length")
plt.ylabel("Accuracy")
plt.title("Model Accuracy by Sequence Length on Test Set")
plt.legend()
plt.savefig("../Figures/length_accuracy_comparison.png")