# Allow imports from parent directory
import os,sys
dir_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(dir_path)
sys.path.append(parent_dir)

from util import *
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(2025)

BATCH_SIZE = 2048
EMBED_DIM = 20
RNN_HIDDEN_DIM = 128
FC_HIDDEN_DIM = 128


train_dataloader, val_dataloader, test_dataloader, voc = load_data(batch_size=BATCH_SIZE) 
loss_fn = torch.nn.CrossEntropyLoss()


num_class = 5
vocab_size = len(voc)
model = RNN(vocab_size, EMBED_DIM, RNN_HIDDEN_DIM, FC_HIDDEN_DIM, num_classes=5).to(device)

# Test accuracy 0.30604461538461536
model.load_state_dict(torch.load("../Models/rnn_model_3E.pth", weights_only=True))
print("Model loaded")

conf_matrix = confusion_matrix(model, test_dataloader, device)

sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5])

acc_test, _ = evaluate(model, loss_fn, test_dataloader, device)
print("Test accuracy", acc_test)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for RNN Model on Test Set")
plt.savefig("../Figures/RNN_confusion_matrix_3E.png")

