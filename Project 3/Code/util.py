# Disable deprecation warnings for torchtext datapipes
import warnings
warnings.filterwarnings("ignore",category=UserWarning,module=r"torchdata\.datapipes")
import torchtext;torchtext.disable_torchtext_deprecation_warning()

import torch.nn as nn
from torch.utils.data.dataset import random_split
from torchtext.datasets import AmazonReviewFull
import re
import torch
from collections import Counter, OrderedDict
from torchtext.vocab import vocab
from torch.utils.data import DataLoader
import sys
from time import perf_counter
import numpy as np


def tokenizer(text):
    """
    Tokenizes text by removing HTML tags, lowercasing, replacing non-word characters with spaces, and appending detected ASCII emoticons (with hyphens removed) as tokens. 
    Splits on whitespace and returns the resulting list.

    Args:
        text (str): The input text to tokenize.

    Returns:
        list[str]: Tokens from the cleaned text plus extracted emoticons.
    """
    text = re.sub(r'<[^>]*>', '', text)
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|d|p)', text.lower())
    text = re.sub(r'[\W]+', ' ', text.lower()) +' '.join(emoticons).replace('-', '')
    tokenized = text.split()
    return tokenized



def load_data(batch_size=32, sample_size=None, use_cache=True, cache_file='Data/data_cache.pth'):
    """
    Loads and prepares the AmazonReviewFull dataset by optionally restoring cached datasets/vocabulary, performing an 80/20 train/validation split, building a frequency-based vocabulary from the training data. 
    Creates DataLoaders with a custom collate function that tokenizes, indexes, pads sequences, and tracks lengths. 
    Supports creating a balanced subset of the data with sample_size, and caches/restores raw splits and vocab when use_cache is True (only reuses cache if sample_size matches).

    Args:
        batch_size (int): Batch size for the DataLoaders.
        sample_size (int | None): If provided, number of training samples to use; test size is set to 20% of this. Uses full dataset when None.
        use_cache (bool): Whether to load/save cached splits and vocabulary.
        cache_file (str): Path to the cache file used for saving/loading.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, Any]: Train, validation, and test DataLoaders, and the constructed vocabulary object.
    """
    # Try to load from cache if it exists
    if use_cache:
        try:
            print("Loading data from cache...")
            cached_data = torch.load(cache_file)
            # Check if sample_size matches
            if cached_data['sample_size'] == sample_size:
                print("Cache loaded successfully! Building dataloaders...")
                train_data = cached_data['train_data']
                val_data = cached_data['val_data']
                testing_dataset = cached_data['test_data']
                voc = cached_data['voc']
                
                # Rebuild collate_batch function
                text_pipeline = lambda x: [voc[token] for token in tokenizer(x)]
                label_pipeline = lambda x: int(x) - 1

                def collate_batch(batch):
                    """
                    Converts a batch of (label, text) pairs into parallel lists of numeric labels, tokenized text tensors, and their lengths, skipping any empty tokenized sequences. 
                    Uses global text_pipeline and label_pipeline` to transform inputs.

                    Args:
                        batch (Iterable[Tuple[Any, str]]): An iterable of (label, text) items to collate.

                    Returns:
                        None
                    """
                    label_list, text_list, lengths = [], [], []
                    for (_label, _text) in batch:
                        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
                        if processed_text.size(0) > 0:
                            label_list.append(label_pipeline(_label))
                            text_list.append(processed_text)
                            lengths.append(processed_text.size(0))
                    
                    label_list = torch.tensor(label_list, dtype=torch.int64)
                    lengths = torch.tensor(lengths, dtype=torch.int64)
                    padded_texts = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=0)
                    return padded_texts, label_list, lengths

                # Rebuild dataloaders
                train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
                val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
                test_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
                
                return train_dataloader, val_dataloader, test_dataloader, voc
            else:
                print("Cache parameters don't match, rebuilding data...")
        except FileNotFoundError:
            print("No cache found, building data from scratch...")
        except Exception as e:
            print(f"Error loading cache: {e}, rebuilding data...")
    
    training_dataset = AmazonReviewFull(split='train')
    testing_dataset = AmazonReviewFull(split='test')

    torch.manual_seed(2025)

    # If sample_size is specified, create a stratified sample
    if sample_size is not None:
        from collections import defaultdict
        
        # Group by label to ensure balanced sampling for training data
        label_data = defaultdict(list)
        for label, text in training_dataset:
            label_data[label].append((label, text))
        
        # Sample proportionally from each class
        sampled_data = []
        samples_per_class = sample_size // len(label_data)
        for label, items in label_data.items():
            sampled_data.extend(items[:samples_per_class])
        
        training_dataset = sampled_data
        
        # Also sample test dataset (use 20% of training sample size)
        test_label_data = defaultdict(list)
        for label, text in testing_dataset:
            test_label_data[label].append((label, text))
        
        test_sampled_data = []
        test_samples_per_class = (sample_size // 5) // len(test_label_data)  # 20% of sample_size
        for label, items in test_label_data.items():
            test_sampled_data.extend(items[:test_samples_per_class])
        
        testing_dataset = test_sampled_data
    
    # Split into train and validation (80/20 split)
    training_dataset = list(training_dataset)
    train_size = int(0.8 * len(training_dataset))
    val_size = len(training_dataset) - train_size
    train_data, val_data = random_split(training_dataset, [train_size, val_size])

    token_counts = Counter()
    for label, line in train_data:
        tokens = tokenizer(line)
        token_counts.update(tokens)

    sorted_by_freq_tuples = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    voc = vocab(ordered_dict)
    voc.insert_token('<pad>', 0)
    voc.insert_token('<unk>', 1)
    voc.set_default_index(1)

    text_pipeline = lambda x: [voc[token] for token in tokenizer(x)]
    label_pipeline = lambda x: int(x) - 1

    def collate_batch(batch):
        """
        Converts a batch of (label, text) pairs into parallel lists of numeric labels, tokenized text tensors, and their lengths, skipping any empty tokenized sequences. 
        Uses global text_pipeline and label_pipeline` to transform inputs.

        Args:
            batch (Iterable[Tuple[Any, str]]): An iterable of (label, text) items to collate.

        Returns:
            None
        """
        label_list, text_list, lengths = [], [], []
        for (_label, _text) in batch:
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            if processed_text.size(0) > 0:  # Only include non-empty sequences
                label_list.append(label_pipeline(_label))
                text_list.append(processed_text)
                lengths.append(processed_text.size(0))
            
        label_list = torch.tensor(label_list, dtype=torch.int64)
        lengths = torch.tensor(lengths, dtype=torch.int64)
        padded_texts = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=0)
        return padded_texts, label_list, lengths

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    # Save to cache if enabled (save raw data, not dataloaders)
    if use_cache:
        print("Saving data to cache...")
        torch.save({
            'train_data': list(train_data),
            'val_data': list(val_data),
            'test_data': list(testing_dataset),
            'voc': voc,
            'sample_size': sample_size
        }, cache_file)
        print("Data cached successfully!")

    return train_dataloader, val_dataloader, test_dataloader, voc

def train(model, optimizer, loss_fn, dataloader, device):
    """
    Trains the model for one epoch over the given dataloader: moves batches to device, performs forward/backward passes with the provided loss_fn and optimizer.
    Tracks running accuracy and loss. Returns dataset-wide average accuracy and loss.

    Args:
        model: The model to train; must support a forward call as model(text_batch, lengths).
        optimizer: Optimizer instance used to update model parameters.
        loss_fn: Loss function taking (predictions, labels) and returning a scalar loss.
        dataloader: Iterable yielding (text_batch, label_batch, lengths) for each batch.
        device: torch.device on which to run computations.

    Returns:
        Tuple[float, float]: (average_accuracy, average_loss) over the entire dataset.
    """
    model.train()
    total_acc, total_loss = 0, 0

    bar = ProgressBar(len(dataloader))

    for text_batch, label_batch, lengths in dataloader:
        text_batch = text_batch.to(device)
        label_batch = label_batch.to(device)
        lengths = lengths.to(device)
        optimizer.zero_grad()
        pred = model(text_batch, lengths)
        loss = loss_fn(pred, label_batch)
        loss.backward()
        optimizer.step()
        total_acc += (pred.argmax(1) == label_batch).sum().item()
        total_loss += loss.item() * label_batch.size(0)
        bar.step()

    bar.finish()
    return total_acc / len(dataloader.dataset), total_loss / len(dataloader.dataset)

def evaluate(model, loss_fn, dataloader, device):
    """
    Evaluates the model on the given dataloader without gradient computation. 
    Runs forward passes, and aggregates dataset-wide average accuracy and loss. 

    Args:
        model: The model to evaluate; should support a forward call as model(text_batch, lengths).
        loss_fn: Loss function taking (predictions, labels) and returning a scalar loss.
        dataloader: Iterable yielding (text_batch, label_batch, lengths) for each batch.
        device: torch.device on which to run computations.

    Returns:
        Tuple[float, float]: (average_accuracy, average_loss) over the entire dataset.
    """
    model.eval()
    total_acc, total_loss = 0, 0

    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            text_batch = text_batch.to(device)
            label_batch = label_batch.to(device)
            lengths = lengths.to(device)
            pred = model(text_batch, lengths)
            loss = loss_fn(pred, label_batch)
            total_acc += (pred.argmax(1) == label_batch).sum().item()
            total_loss += loss.item() * label_batch.size(0)

    return total_acc / len(dataloader.dataset), total_loss / len(dataloader.dataset)

def confusion_matrix(model, dataloader, device, num_classes=5):
    """
    Computes a confusion matrix over the dataloader, accumulating counts of true (rows) vs. predicted (columns) labels into a num_classes × num_classes matrix. 

    Args:
        model: Trained model; should support a forward call as model(text_batch, lengths).
        dataloader: Iterable yielding (text_batch, label_batch, lengths) for each batch.
        device: torch.device on which to run computations.
        num_classes (int): Number of classes; determines the size of the square confusion matrix.

    Returns:
        numpy.ndarray: An (num_classes, num_classes) integer matrix where entry [i, j] is the count of samples with true label i predicted as j.
    """

    model.eval()
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            text_batch = text_batch.to(device)
            label_batch = label_batch.to(device)
            lengths = lengths.to(device)
            pred = model(text_batch, lengths)
            predicted_labels = pred.argmax(1)
            for t, p in zip(label_batch.view(-1), predicted_labels.view(-1)):
                conf_matrix[t.long(), p.long()] += 1

    return conf_matrix

def count_parameters(model):
    """
    Return the number of trainable parameters in the given PyTorch model.

    Args: 
        model (torch.nn.Module): The PyTorch model whose trainable parameters will be counted.

    Returns: 
        int: Total number of parameters with requires_grad=True. 
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def compute_length_accuracy(model, dataloader, device):
    """
    Compute accuracy of a PyTorch classification model binned by input length (0-50, 50-100, ...).

    Args:
        model (torch.nn.Module): The model used to produce predictions.
        dataloader (Iterable): Yields batches of (text, labels, lengths).
        device (torch.device): Device to move inputs to before inference.

    Returns:
        tuple: (bins, accuracies) where `bins` is a list of bin start lengths (0, 50, 100, ...)
            and `accuracies` is a list of corresponding accuracy percentages (floats in [0,1]).
    """
    correct_by_len = {} # {length_bin: count}
    total_by_len = {}
    
    model.eval()
    with torch.no_grad():
        for text, labels, lengths in dataloader:
            text, labels = text.to(device), labels.to(device)
            preds = model(text, lengths).argmax(1)
            correct = (preds == labels).cpu().numpy()
            lengths = lengths.cpu().numpy()
            
            for is_correct, length in zip(correct, lengths):
                # Bin lengths: 0-50, 50-100, etc.
                bin_start = (length // 50) * 50
                if bin_start not in total_by_len:
                    total_by_len[bin_start] = 0
                    correct_by_len[bin_start] = 0
                total_by_len[bin_start] += 1
                correct_by_len[bin_start] += is_correct

    # Calculate percentages
    bins = sorted(total_by_len.keys())
    accuracies = [correct_by_len[b] / total_by_len[b] for b in bins]
    return bins, accuracies

class RNN(nn.Module):
    """
    A simple text classification RNN that embeds token indices, processes sequences with a vanilla RNN,
    and maps the final hidden state through a small MLP to produce class class scores. Handles variable-length
    padded batches via pack_padded_sequence (expects padding_idx=0 and lengths for each sequence).

    Args:
        vocab_size (int): Size of the vocabulary (number of unique token indices).
        embed_dim (int): Dimension of the embedding vectors.
        rnn_hidden_size (int): Hidden size of the RNN layer.
        fc_hidden_size (int): Hidden size of the intermediate fully connected layer.
        num_classes (int): Number of output classes (size of the final class scores).
    """
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.output_activation = nn.Linear(fc_hidden_size, num_classes)
    
    def forward(self, text, lengths):
        """
        Runs a forward pass: embeds padded token sequences, packs by lengths, feeds through the RNN,
        takes the final hidden state, applies a fully connected layer with ReLU, and outputs class class scores.

        Args:
            text (torch.Tensor): Padded token indices of shape (batch_size, seq_len) with padding_idx=0.
            lengths (torch.Tensor): True sequence lengths for each item, shape (batch_size,).

        Returns:
            torch.Tensor: Class scores of shape (batch_size, num_classes).
        """
        embedded = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, hidden = self.rnn(out)
        out = hidden.squeeze(0)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.output_activation(out)
        return out


class LSTM(nn.Module):
    """
    A text classification model using embeddings followed by an LSTM encoder and a small MLP head
    to produce class class scores. Supports variable-length, padded batches via pack_padded_sequence
    (expects padding_idx=0 and per-sequence lengths).

    Args:
        vocab_size (int): Size of the vocabulary (number of token indices).
        embed_dim (int): Dimension of embedding vectors.
        rnn_hidden_size (int): Hidden size of the LSTM.
        fc_hidden_size (int): Hidden size of the fully connected layer before the output.
        num_classes (int): Number of output classes.
    """
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.output_activation = nn.Linear(fc_hidden_size, num_classes)
    
    def forward(self, text, lengths):
        """
        Forward pass: embeds input tokens, packs by lengths, encodes with LSTM, takes the final
        hidden state, and maps through a fully connected layer with ReLU to class class scores.

        Args:
            text (torch.Tensor): Padded token indices of shape (batch_size, seq_len) with padding_idx=0.
            lengths (torch.Tensor): True lengths for each sequence, shape (batch_size,).

        Returns:
            torch.Tensor: Class scores of shape (batch_size, num_classes).
        """
        embedded = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, (hidden, cell) = self.rnn(out)
        out = hidden.squeeze(0)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.output_activation(out)
        return out

class BoW(nn.Module):
    """
    A simple bag-of-words classifier that averages token embeddings with nn.EmbeddingBag
    and feeds the pooled representation to a linear layer to produce class class scores.

    Args:
        vocab_size (int): Size of the vocabulary (number of token indices).
        embed_dim (int): Dimension of the embedding vectors.
        num_class (int): Number of output classes.
    """
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True, mode='mean', padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        """
        Initializes embedding and linear layer weights uniformly within [-0.5, 0.5]
        and zeros the linear layer bias to stabilize early training.
        """
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, lenghts):
        """
        Computes class class scores by pooling token embeddings with EmbeddingBag (mean)
        and passing the pooled vector through a linear classifier. The 'lenghts'
        argument is accepted but not used in this implementation.

        Args:
            text (torch.Tensor): Token indices; shape depends on EmbeddingBag usage.
            lenghts (torch.Tensor): Sequence lengths (unused).

        Returns:
            torch.Tensor: Class scores of shape (batch_size, num_class).
        """
        embedded = self.embedding(text)
        return self.fc(embedded)


class ProgressBar:
    """
    A simple console progress bar that displays progress and estimated remaining time.
    """
    def __init__(self, width):
        self.width = min(width, 50)
        self.scale = self.width / width
        self.pos = 0
        self.steps = 0
        self.total_steps = width
        self.time = perf_counter()
    def step(self):
        self.steps+=1
        if int(self.steps * self.scale)>self.pos:
            self.pos+=1
            time_rem = (perf_counter() - self.time) * (self.total_steps - self.steps) / self.steps
            bar = "[" + "#" * self.pos + "-" * (self.width - self.pos) + "] " + f"{time_rem:.2f}s remaining    "
            sys.stdout.write("\r" + bar)
            sys.stdout.flush()
    def finish(self):
        bar = "[" + "#" * self.width + "]\n"
        sys.stdout.write("\r" + bar)
        sys.stdout.flush()
        