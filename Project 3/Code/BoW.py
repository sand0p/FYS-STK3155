from util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(2025)

# Model parameters
BATCH_SIZE = 2048
EMBED_DIM = 64
EPOCHS = 3
LEARNING_RATE = 4.0
NUM_CLASSES = 5

# Loading data
train_dataloader, val_dataloader, test_dataloader, voc = load_data(batch_size=BATCH_SIZE) 
vocab_size = len(voc)

model = BoW(vocab_size, EMBED_DIM, NUM_CLASSES).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

with open("Models/training_log.txt", "a") as f:
    f.write(f"BoW\n[Epoch, Train accuracy, Train loss, Validation accuracy, Validation loss]\n")
beg_tm = perf_counter()

# Model training
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    acc_train, loss_train = train(model, optimizer, loss_fn, train_dataloader, device)
    acc_val, loss_acc = evaluate(model, loss_fn, val_dataloader, device)
    print("Train accuracy",acc_train)
    print("Validation accuracy",acc_val)

    # Add accuracy and loss to external file
    with open("Models/training_log.txt", "a") as f:
        f.write(f"[{epoch}, {acc_train}, {loss_train}, {acc_val}, {loss_acc}]\n")
print("Total time", perf_counter()-beg_tm)

# Saving model
torch.save(model.state_dict(), 'Models/bow_model_SGD_3E.pth')
print("Model saved to Models/bow_model_SGD_3E.pth")

print("Checking test set accuracy")
acc_test, loss_test = evaluate(model, loss_fn, test_dataloader, device)
print(f'Test accuracy: {acc_test:8.3f}')
print(f'Test loss: {loss_test:8.3f}')

print("Number of parameters:", count_parameters(model))