# Sentiment analysis of Amazon reviews using PyTorch
## A comparison of traditional and deep learning models
By Sander Oppen and Ole Markus Bråthen

We perform sentiment review with multiclass classification of amazon reviews from the `AmazonReviewFull` dataset. We find that LSTM models  outperform traditional BoW and RNN models, with a $61\%$ accuracy rate on the test set.

The folders in Code contain:

- Figures: Figures used in the report
- Plotting: Code for evaluating models, plotting confusion matrices, and all other plots in Figures
- Models: The state dicts for the models produced
- LLM: Conversations with LLMs used in the creation of the report

The default directory for data caches is set to a folder "Data". The local caches are not included in the repo, as the filesize (1.5GB) exceeds the GitHub storage limit.

All code is written for Python 3.12.12. All packages required can be installed with:
```
pip install -r requirements.txt
```

### Notes on usage
In `amazonreviewfull.py` in the torchtext library files, replace the URL with 
```
https://drive.usercontent.google.com/download?id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA&export=download
```

When working on UiO HPC servers, ensure that the modules `libffi`, `bzip2`, and `OpenSSL` are loaded, as they are required for processing the dataset.