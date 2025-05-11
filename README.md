# Emoji-Aware Hate Speech Detection with BERT Model

## Overview

This project explores whether incorporating emoji information improves hate speech detection in online comments. Emojis often carry nuanced emotional or contextual cues that 
are not captured by plain text models. Our goal was to evaluate whether enriching input with emoji semantics improves the performance of deep learning models, specifically BERT, for hate speech classification.

## Dataset

We used the `final_hateXplain.csv` dataset from Kaggle (https://www.kaggle.com/datasets/sayankr007/cyber-bullying-data-for-multi-label-classification/data?select=final_hateXplain.csv), 
which includes social media comments labeled into three classes:
- **0**: Normal
- **1**: Hate Speech
- **2**: Offensive

There were two files available on Kaggle, `hateXplain.csv`, which is the original dataset, and `final_hateXplain.csv`, which is the processed and labelled version of the 
original dataset. We originally began by using the `hateXplain.csv` dataset when we were building our experimental CNN model, but once we decided to switch to using a BERT 
model for this project, we decided that it would be make more sense to to use the `final_hateXplain.csv` file, since it contains the labels of the classes in the original 
dataset, which we needed to train and test our model on. 

To focus on our objective, we filtered the `final_hateXplain.csv` dataset to only include comments that contained at least one emoji.

## Models

We implemented and compared two types of BERT models:
1. **Text-Only Model**: Uses only the raw text of the comment, and removes emojis from the input as part of the preprocessing step.
2. **Emoji-Aware Model**: Converts emojis into their textual descriptions (e.g., "😂" → "face with tears of joy") using the `emoji2desc_dedup` function, which also simplifies
the emoji conversion by getting rid of repeated emoji descriptions if there is more than one of the same emoji in the comment, to avoid adding unneccessary noise to the input.

### BERT-Based Models
- Fine-tuned `bert-base-uncased` using a custom PyTorch training loop
- Used a stratified 80/20 train-test split, to maintain class proportions in both training and test sets
- Tracked training/testing loss and accuracy across epochs
- Evaluated with multiple metrics, including precision, recall, F1-score, accuracy (including macro average and weighted average accuracy), and a confusion matrix

### CNN-Based Models
- A simple Conv1D + max-pooling architecture with an LSTM layer and GloVe embeddings.
- The CNN results were incomplete due to earlier errors, and are not part of our final conclusions.

## Results Summary

Our results showed that emoji-aware BERT performed performs slightly better than the baseline (text-only) model, but only by a small margin. We believe this is due to the fact 
that our dataset did not contain a signficant amount of emojis in the input, even though we filtered our dataset to contain only comments that include emojis to truly test the 
impact of emoji sentiment in online hate speech detection. Thus, the text + emoji model may have more of a signficant impact when used on datasets that have more emoji-heavy input.

## Visualizations

- Class distribution in the dataset
- Number of Emojis per Comment chart
- Train/Test accuracy and loss over epochs
- Confusion matrices
- Precision/Recall/F1/Accuracy tables


## How to Run

This project was implemented in **Google Colab** using:
- `transformers`
- `torch`
- `emoji`
- `sklearn`
- `matplotlib`, `seaborn`

To run the notebook:
1. Download the `final_hateXplain.csv` dataset on your computer and upload it to the files tab of Google Colab
2. Run cells in order to preprocess, train both models, and evaluate
3. both BERT models are saved at the end to the files tab if needed for later use

## Authors

This project was completed by 3 group members, Daphne Ozkan, Alice Yang, and Boyoon Han as our Group Project for DS-UA 301: Advanced Techniques in Machine Learning 
and Deep Learning.

## Notes

- The `Experimental_CNN_Model_Not_Final.ipynb` file uses the unprocessed `hateXplain.csv` dataset and does not reflect our final methodology.
- It is provided for reference and transparency only, as mentioned in our presentation.
