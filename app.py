from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import datetime
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load Hugging Face sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    sentiment = sentiment_pipeline(review)
    return jsonify({'sentiment': sentiment[0]['label'], 'confidence': sentiment[0]['score']})
# Business data - Placeholder for tracking reviews and sentiment
review_data = []


# Function to analyze sentiment for customer feedback
def analyze_sentiment(text):
    sentiment = sentiment_pipeline(text)
    label = sentiment[0]['label']
    confidence = sentiment[0]['score']
    return label, confidence


# Route to render the home page
@app.route('/')
def index():
    return render_template('index.html')


# Route to handle review submission
@app.route('/submit_review', methods=['POST'])
def submit_review():
    review = request.form['review']
    sentiment, confidence = analyze_sentiment(review)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add review data to review_data
    review_data.append({
        'review': review,
        'sentiment': sentiment,
        'confidence': confidence,
        'timestamp': timestamp
    })

    # Send feedback to user
    return jsonify({'sentiment': sentiment, 'confidence': confidence, 'timestamp': timestamp})


# Route to display business analytics dashboard
@app.route('/dashboard')
def dashboard():
    # Prepare data for visualization
    sentiments = [review['sentiment'] for review in review_data]
    sentiment_counts = pd.Series(sentiments).value_counts()

    # Create sentiment distribution chart
    fig, ax = plt.subplots(figsize=(6, 4))
    sentiment_counts.plot(kind='bar', ax=ax, color=['green', 'yellow', 'red'])
    ax.set_title('Sentiment Distribution of Customer Reviews')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')

    # Save plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    return render_template('dashboard.html', sentiment_img=img_base64, sentiment_counts=sentiment_counts)


if __name__ == "__main__":
    app.run(debug=True)
