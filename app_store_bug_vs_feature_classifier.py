import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Training data
data = {
    "review": [
        "App keeps crashing",
        "The app crashes when I open settings",
        "The application is not working",
        "I get an error when I try to login",
        "The app freezes frequently",
        "Please add dark mode",
        "Please add a search feature",
        "It would be great to include a notification option",
        "Please provide a better filter",
        "Can you add more customization options?"
    ],
    "label": [
        "Bug",
        "Bug",
        "Bug",
        "Bug",
        "Bug",
        "Feature Request",
        "Feature Request",
        "Feature Request",
        "Feature Request",
        "Feature Request"
    ]
}

df = pd.DataFrame(data)

# Build TF-IDF + Naive Bayes pipeline
model = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)

# Train the model
model.fit(df["review"], df["label"])


def classify_review(user_input):
    """Classify a review as Bug or Feature Request."""
    prediction = model.predict([user_input])[0]
    return prediction


# Take user input and classify it
print("App Store Bug vs Feature Request Classifier")
print("Type 'exit' to stop.")

while True:
    user_text = input("\nEnter an app review: ")

    if user_text.lower() == "exit":
        print("Exiting...")
        break

    result = classify_review(user_text)
    print(f"Result: The review is classified as [{result.upper()}]")
