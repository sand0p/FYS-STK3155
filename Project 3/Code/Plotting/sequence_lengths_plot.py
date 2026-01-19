# Allow imports from parent directory
import os,sys
dir_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(dir_path)
sys.path.append(parent_dir)

from util import *
import matplotlib.pyplot as plt
train_loader, _, _, _ = load_data(batch_size=2048, use_cache=True)
print("loaded data")
all_lengths = []

for _,_,lengths in train_loader:
    all_lengths.extend(lengths.tolist())

plt.figure(dpi=300)
plt.hist(all_lengths, bins=100, range=(0,225), edgecolor='black')
plt.title("Distribution of sequence lengths in training set")
plt.xlabel("Sequence length")
plt.ylabel("Count")
plt.savefig("../Figures/sequence_length_distribution_no_log.png")