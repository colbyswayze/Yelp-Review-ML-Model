# Colby Swayze - Texas A&M University - 1/10/2025

# Licensing and Attribution
# This project uses the Yelp Academic Dataset.
# Due to licensing restrictions, the dataset is not included in this repository.
# You can download it from the Yelp Dataset website: https://www.yelp.com/dataset
# This code is for educational and personal use only and is subject to the terms of the Yelp Academic Dataset license.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import os

# Step 1: Load Dataset in Chunks
def load_data_in_chunks(source, chunksize=100000, max_chunks=1): # I am only usinh aproximately 0.1G of data for the model to keep processing under 10 minutes.
    """Load dataset from a file path in chunks."""
    print("Loading dataset in chunks...")
    chunks = []
    for i, chunk in enumerate(pd.read_json(source, lines=True, chunksize=chunksize)):
        if i >= max_chunks:
            break  # Stop after loading the specified number of chunks
        # Check for required columns
        if 'stars' in chunk.columns and 'text' in chunk.columns:
            chunks.append(chunk[['stars', 'text']])  # Keep only relevant columns
        else:
            print(f"Required columns 'stars' or 'text' not found in the chunk. Available columns: {chunk.columns.tolist()}")
            raise KeyError("The dataset does not have the required columns: 'stars' and 'text'.")
        print(f"Loaded chunk with {chunk.shape[0]} rows.")
    data = pd.concat(chunks, ignore_index=True)
    print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
    return data

# Step 1.1: Generate Dummy Dataset
# I am including this step to show how the regression model works in small constraints in the case the file cannot be found for legitimate training. However in this case it is purely hardcoded data, no processing.
def generate_dummy_data():
    """Generate a small dummy dataset for testing."""
    print("Generating dummy dataset...")
    data = pd.DataFrame({
        'stars': [5, 1, 3, 5, 1, 3, 5, 2, 3, 5],
        'text': [
            "This product is amazing! Highly recommend it.",
            "Terrible quality, not worth the money.",
            "It's okay, does the job but nothing special.",
            "Fantastic value for the price!",
            "Worst purchase ever. Completely disappointed.",
            "Decent product, but could be improved.",
            "I absolutely love this! Five stars.",
            "Not what I expected, quite disappointing.",
            "Satisfactory, works as described.",
            "Exceeded my expectations! Great buy."
        ]
    })
    print(f"Dummy dataset created with {len(data)} rows.")
    return data

# Step 2: Preprocess Data
# This step provides counts for each sentament and a bar graph of the disributed catagories using MATPLOTLIB
def preprocess_data(data):
    """Clean and preprocess the review data using stars for sentiment."""
    print("Preprocessing data...")

    # Map star ratings to sentiment labels
    def map_sentiment(stars):
        if stars >= 4:
            return 'Positive'
        elif stars == 3:
            return 'Neutral'
        else:
            return 'Negative'

    data['sentiment'] = data['stars'].apply(map_sentiment)

    # Count and print sentiment distribution
    sentiment_counts = data['sentiment'].value_counts()
    print("Sentiment Counts:")
    print(sentiment_counts)

    # Plot sentiment distribution
    print("Visualizing sentiment distribution...")
    plt.figure(figsize=(8, 5))
    sentiment_counts.plot(kind='bar', color=['green', 'blue', 'red'])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.savefig("sentiment_distribution.png")  # Save the plot
    plt.show()
    print("Sentiment distribution visualization saved as 'sentiment_distribution.png'.")

    print("Preprocessing complete.")
    return data

# Step 3: Train Model
def train_model(data):
    """Train a Logistic Regression model on the preprocessed data."""
    print("Training model...")

    # Feature extraction using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['text'])
    y = data['sentiment']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    print("Model training complete.")
    return model, vectorizer, X_test, y_test

# Step 4: Evaluate Model
def evaluate_model(model, vectorizer, X_test, y_test):
    """Evaluate the trained model on test data."""
    print("Evaluating model...")

    # Make predictions
    predictions = model.predict(X_test)

    # Generate and display classification report
    report = classification_report(y_test, predictions)
    print("Evaluation complete. Classification Report:")
    print(report)

    # Confusion matrix visualization
    print("Visualizing confusion matrix...")
    cm = confusion_matrix(y_test, predictions, labels=['Positive', 'Neutral', 'Negative'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Positive', 'Neutral', 'Negative'], 
                yticklabels=['Positive', 'Neutral', 'Negative'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()
    print("Confusion matrix visualization displayed")

# Main Execution
# This function runs the main program functions and also includes try-except statements to keep the user console updated during model run time.
if __name__ == "__main__":
    # Specify dataset source (local file path)
    dataset_source = "C:/Users/admin/Downloads/yelp_dataset/yelp_academic_dataset_review.json"

    # Debug: Check if file exists
    print(f"Looking for dataset at: {dataset_source}")
    if not os.path.exists(dataset_source):
        raise FileNotFoundError(f"Dataset file '{dataset_source}' does not exist.")

    try:
        print("Loading dataset...")
        data = load_data_in_chunks(dataset_source)
        print(f"Dataset loaded successfully with {len(data)} rows.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using dummy dataset instead.")
        data = generate_dummy_data()

    try:
        print("Starting preprocessing...")
        preprocessed_data = preprocess_data(data)
        print("Preprocessing complete.")

        print("Starting model training...")
        model, vectorizer, X_test, y_test = train_model(preprocessed_data)
        print("Model training complete.")

        print("Starting evaluation...")
        evaluate_model(model, vectorizer, X_test, y_test)
        print("Evaluation complete.")
    except Exception as e:
        print(f"Error during processing: {e}")
