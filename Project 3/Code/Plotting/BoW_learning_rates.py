# Allow imports from parent directory
import os,sys
dir_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(dir_path)
sys.path.append(parent_dir)

from util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(2025)

BATCH_SIZE = 2048
EMBED_DIM = 64
EPOCHS = 3
LEARNING_RATE = 4.0

train_dataloader, val_dataloader, test_dataloader, voc = load_data(batch_size=BATCH_SIZE) 

num_class = 5
vocab_size = len(voc)

accuracy = []
learning_rates = [0.01,0.05,0.1,0.5,1,2,3,4,5,6]
for lr in learning_rates:
    model = BoW(vocab_size, EMBED_DIM, num_class).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS} with learning rate {lr}")
        acc_train, loss_train = train(model, optimizer, loss_fn, train_dataloader, device)
        acc_val, loss_acc = evaluate(model, loss_fn, val_dataloader, device)
        print("Train accuracy",acc_train)
        print("Validation accuracy",acc_val)
    
    acc_test, loss_test = evaluate(model, loss_fn, test_dataloader, device)
    print(f"Test accuracy with learning rate {lr}: {acc_test:8.3f}")
    accuracy.append((lr, acc_test))
accuracy.sort()
print(accuracy)

import seaborn as sns
import matplotlib.pyplot as plt

acc = [a[1] for a in accuracy]
lr = [a[0] for a in accuracy]
plt.figure(dpi = 300)
sns.lineplot(x=lr, y=acc, marker="o", label="Test Accuracy")
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Test Accuracy")
plt.title("Learning Rate vs Test Accuracy for BoW Model")
plt.legend()
plt.savefig("../Figures/learning_rate_vs_accuracy.png")    