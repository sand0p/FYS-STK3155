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
EMBED_DIM = 64
EPOCHS = 50
LEARNING_RATE = 4.0

train_dataloader, val_dataloader, test_dataloader, voc = load_data(batch_size=BATCH_SIZE) 
loss_fn = torch.nn.CrossEntropyLoss()


num_class = 5
vocab_size = len(voc)
model = BoW(vocab_size, EMBED_DIM, num_class).to(device)

# Test accuracy 0.525576923076923
model.load_state_dict(torch.load("../Models/bow_model_SGD_3E.pth", weights_only=True))
print("Model loaded")

conf_matrix = confusion_matrix(model, test_dataloader, device)
acc_test, _ = evaluate(model, loss_fn, test_dataloader, device)
print("Test accuracy", acc_test)

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for BoW Model on Test Set")
plt.savefig("../Figures/bow_confusion_matrix_3E.png")