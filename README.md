# BERT Sentiment Analysis for Product Reviews

This project implements sentiment analysis on product reviews using BERT (Bidirectional Encoder Representations from Transformers). It involves data preprocessing, BERT model training, evaluation, and predicting sentiment on new reviews.

## Overview

Sentiment analysis is performed to classify product reviews into positive or negative sentiments using state-of-the-art deep learning techniques with BERT. The project includes two main parts:

1. **Data Preprocessing:**
   - Initial data exploration and cleaning using pandas and regex.
   - Handling duplicates and ensuring balanced classes for training.

2. **BERT Model Training and Evaluation:**
   - Utilizing the `transformers` library to fine-tune a BERT model for sentiment classification.
   - Training the model on a subset of reviews and evaluating its performance using accuracy metrics and a confusion matrix.
   - Saving and loading the trained model for deployment.

3. **Predicting Sentiment:**
   - Providing a function to predict sentiment on new reviews using the trained BERT model.

## Files Included

- `data cleaning.ipynb`: Python script for cleaning the raw dataset.
- `bert sentiment analysis.ipynb`: Python script for training the BERT model and evaluating its performance.
- `README.md`: This file, containing project overview, instructions, and documentation.
- `requirements.txt`: List of Python dependencies required to run the scripts.
- `Dataset Used.txt`: Link of the amazon reviews dataset.

## Usage

1. **Setup:**
   - Clone the repository: `git https://github.com/youssef00mohamed/BERT-Based-Sentiment-Analysis-Model-for-Product-Reviews`
   - Install dependencies: `pip install -r requirements.txt`

2. **Data Cleaning:**
   - Run `data cleaning.ipynb` to preprocess the raw dataset (`train.csv`).
   - Cleaned dataset will be saved as `CleanedTrain.csv`.

3. **BERT Model Training:**
   - Adjust parameters such as `batch_size`, `num_epochs`, and `learning_rate` in `bert sentiment analysis.ipynb` if needed.
   - Run `bert sentiment analysis.ipynb` to train and evaluate the BERT model.
   - Trained model will be saved as `NEW_BERT.pth`.

4. **Predicting Sentiment:**
   - Use the `predict_sentiment()` function in `bert sentiment analysis.ipynb` to predict sentiment on new reviews.

## Requirements

Ensure you have Python 3.x installed along with the following libraries:

- `pandas`
- `numpy`
- `nltk`
- `scikit-learn`
- `torch`
- `transformers`

Install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```

## Acknowledgments

  - This project uses the BERT model from the Hugging Face `transformers` library.
  - Dataset sourced from [https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews].

## Contributions

Contributions to this repository are welcome! If you have suggestions for improvements, additional compression algorithms, or enhancements to the GUI application, please feel free to open an issue or pull request. Ensure that your contributions align with the project's objectives and maintain code quality and documentation standards.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
