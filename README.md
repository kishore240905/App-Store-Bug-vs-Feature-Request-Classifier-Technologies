# App Store Bug vs Feature Request Classifier

A mini machine learning project that classifies App Store reviews into two categories: **Bug** or **Feature Request**.

## Project Overview

The project uses natural language processing (NLP) and machine learning to analyze user reviews. The review text is converted into numerical features using **TF-IDF**, and a **Multinomial Naive Bayes** classifier predicts the category.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Multinomial Naive Bayes
- Machine Learning / NLP

## How It Works

1. A small labeled dataset of app reviews is prepared.
2. Each review is labeled as either `Bug` or `Feature Request`.
3. TF-IDF converts the review text into numerical features.
4. A pipeline combines TF-IDF with the Multinomial Naive Bayes classifier.
5. The model is trained using the review text and corresponding labels.
6. A user enters a new app review.
7. The trained model predicts whether the review is a **Bug** or **Feature Request**.

### Example

- `App keeps crashing` → **Bug**
- `Please add dark mode` → **Feature Request**

## Project Outcome

The system automatically classifies user reviews as **Bug** or **Feature Request**. This can help organize user feedback and make it easier to identify problems and improvement suggestions.

## My Contribution

This project was completed as a mini project during my Python internship. I worked on preparing the training data, implementing TF-IDF text processing, building the Naive Bayes classification pipeline, training the model, and testing it with new reviews.

## Note

This is an internship mini project created for learning and practical exposure to NLP and text classification.
