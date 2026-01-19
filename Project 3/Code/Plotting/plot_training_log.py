# Allow imports from parent directory
import os,sys
dir_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(dir_path)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
txt = open("../Models/training_log.txt").read().split("\n")
i=2
l1 = txt[0]
lstm = []

EPOCHS = 50

for j in range(EPOCHS):
    lstm.append(eval(txt[i]))
    i+=1
lstm = np.array(lstm)
l2 = txt[i]
i+=2
rnn = []
for j in range(EPOCHS):
    rnn.append(eval(txt[i]))
    i+=1
rnn = np.array(rnn)
l3 = txt[i]
i+=2
bow = []
for j in range(EPOCHS):
    bow.append(eval(txt[i]))
    i+=1
bow=np.array(bow)
plt.figure(dpi = 300)
plt.title("Loss values for different models")
plt.plot(lstm[:,0], lstm[:,2], "r-", label = l1+" training")
plt.plot(lstm[:,0], lstm[:,4], "r--", label = l1+" validation")

plt.plot(rnn[:,0], rnn[:,2], "b-", label = l2+" training")
plt.plot(rnn[:,0], rnn[:,4], "b--", label = l2+" validation")

plt.plot(bow[:,0], bow[:,2], "g-", label = l3+" training")
plt.plot(bow[:,0], bow[:,4], "g--", label = l3+" validation")

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.legend(bbox_to_anchor=(0.67, 0.43), loc="upper left")
plt.tight_layout()

plt.savefig("../Figures/training_plot.png")


plt.figure(2, dpi = 300)
plt.title("Accuracy for different models")
plt.plot(lstm[:,0], lstm[:,1], "r-", label = l1+" training")
plt.plot(lstm[:,0], lstm[:,3], "r--", label = l1+" validation")
plt.plot(rnn[:,0], rnn[:,1], "b-", label = l2+" training")
plt.plot(rnn[:,0], rnn[:,3], "b--", label = l2+" validation")
plt.plot(bow[:,0], bow[:,1], "g-", label = l3+" training")
plt.plot(bow[:,0], bow[:,3], "g--", label = l3+" validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(bbox_to_anchor=(0.67, 0.9), loc="upper left")
plt.tight_layout()
plt.savefig("../Figures/accuracy_plot.png")