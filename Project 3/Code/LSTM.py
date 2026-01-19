from util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(2025)

# Model parameters
BATCH_SIZE = 2048
EMBED_DIM = 20
LSTM_HIDDEN_DIM = 128
FC_HIDDEN_DIM = 128
EPOCHS = 3
NUM_CLASSES = 5
LEARNING_RATE = 0.005

# Loading data
train_dl, valid_dl, test_dl, voc = load_data(batch_size=BATCH_SIZE) 
vocab_size = len(voc)

model = LSTM(vocab_size, EMBED_DIM, LSTM_HIDDEN_DIM, FC_HIDDEN_DIM, num_classes=NUM_CLASSES).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

with open("training_log.txt", "a") as f:
    f.write(f"LSTM\n[Epoch, Train accuracy, Train loss, Validation accuracy, Validation loss]\n")
beg_tm = perf_counter()

# Model training
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    acc_train, loss_train = train(model, optimizer, loss_fn, train_dl, device)
    acc_val, loss_acc = evaluate(model, loss_fn, valid_dl, device)
    print("Train accuracy", acc_train)
    print("Validation accuracy", acc_val)

    # Add accuracy and loss to external file
    with open("training_log.txt", "a") as f:
        f.write(f"[{epoch}, {acc_train}, {loss_train}, {acc_val}, {loss_acc}]\n")
print("Total time", perf_counter()-beg_tm)

# Saving model
torch.save(model.state_dict(), "Models/LSTM_model_3E.pth")
print("Model saved to Models/LSTM_model_3E.pth")

print("Checking test set accuracy")
acc_test, loss_test = evaluate(model, loss_fn, test_dl, device)
print("Test accuracy", acc_test)
print("Test loss", loss_test)
print("Number of parameters:", count_parameters(model))