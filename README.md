# Suicide Ideation Detection Using Tweets

This project involves detecting potential suicide ideation in tweets using Natural Language Processing (NLP) techniques. The model is trained to classify tweets as either a **"Not Suicide post"** or a **"Potential Suicide post"** using the **DistilBERT** transformer model.

The dataset used for training is sourced from [Kaggle: Suicidal Tweet Detection Dataset](https://www.kaggle.com/datasets/aunanya875/suicidal-tweet-detection-dataset).

## Project Overview

The main objective of this project is to train a machine learning model that can automatically classify tweets related to potential suicide ideation. We utilize the following steps to process and model the data:

1. **Data Loading**: The dataset is loaded and cleaned.
2. **Text Preprocessing**: Tweets are cleaned by removing unnecessary characters, links, and stopwords.
3. **Text Tokenization**: We tokenize the text into individual words and filter out rare words.
4. **Model Training**: The pre-trained **DistilBERT** model is fine-tuned to classify the tweets.
5. **Model Evaluation**: The model is evaluated using metrics such as accuracy and F1 score.
6. **SHAP Analysis**: SHAP (SHapley Additive exPlanations) is used to interpret the predictions made by the model.

## Dataset

The dataset is sourced from Kaggle and contains tweets labeled as either **0 (Not Suicide post)** or **1 (Potential Suicide post)**.

- **Dataset Source**: [Suicidal Tweet Detection Dataset](https://www.kaggle.com/datasets/aunanya875/suicidal-tweet-detection-dataset)
- **Columns**:
  - `tweet`: The tweet text.
  - `suicide`: The label for the tweet. `0` for non-suicidal tweets and `1` for suicidal posts.

## Requirements

To run this project, you need the following Python libraries:

- `nltk`
- `numpy`
- `pandas`
- `shap`
- `datasets`
- `transformers`
- `scikit-learn`
- `neattext`
- `matplotlib`

## Project Workflow

### 1. **Data Loading and Preprocessing**

The dataset is loaded from a CSV file, cleaned, and tokenized. The preprocessing includes:

- Removing punctuation, stopwords, URLs, emails, etc.
- Tokenizing text into words and filtering out words that appear too infrequently.

### 2. **Model Training**

We load the **DistilBERT** model, a smaller and more efficient variant of BERT, which is pre-trained for sequence classification tasks.

- **Model**: `distilbert-base-uncased`
- **Training Arguments**:
  - Learning rate: `2e-5`
  - Epochs: `10`
  - Batch size: `8`
  - Early stopping: If no improvement is seen for 3 epochs

### 3. **Model Evaluation**

The trained model is evaluated on the validation set, and metrics like **accuracy** and **F1 score** are calculated.

### 4. **SHAP (SHapley Additive exPlanations)**

SHAP is used to explain the model's predictions. This allows us to see which words in the tweets contribute most to the model’s decision on whether the tweet is a suicide-related post.

### 5. **Inference and Testing**

The model is used to make predictions on the test set. The output includes both true labels and predicted labels for each tweet.

### 6. **Visualization**

A SHAP text plot is used to visualize which parts of the tweet were important for the prediction of "Potential Suicide post."

## Acknowledgements

- The dataset used in this project is from Kaggle: [Suicidal Tweet Detection Dataset](https://www.kaggle.com/datasets/aunanya875/suicidal-tweet-detection-dataset).
- The model used in this project is **DistilBERT**, a smaller, faster, and lighter version of BERT.
- The SHAP library is used for model interpretation.