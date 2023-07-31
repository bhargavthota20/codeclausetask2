import os
import cv2
import pytesseract
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

def extract_text_from_cv_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray_image)
    return text

def perform_sentiment_analysis(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores

def personality_analysis(sentiment_scores):
    compound_score = sentiment_scores["compound"]
    positive_score = sentiment_scores["pos"]
    negative_score = sentiment_scores["neg"]
    personality_traits = []
    if compound_score >= 0.05:
        personality_traits.append("Positive")
    elif compound_score <= -0.05:
        personality_traits.append("Negative")
    else:
        personality_traits.append("Neutral")

    if positive_score > negative_score:
        personality_traits.append("Optimistic")
    elif positive_score < negative_score:
        personality_traits.append("Pessimistic")

    return personality_traits

def main():
    image_filename = input("Enter the name of the CV image file (e.g., cv1.jpg): ")

    script_directory = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_directory, image_filename)

    try:
        
        extracted_text = extract_text_from_cv_image(image_path)

        sentiment_scores = perform_sentiment_analysis(extracted_text)

        personality_traits = personality_analysis(sentiment_scores)

        print("\nPersonality Traits:")
        for trait in personality_traits:
            print(trait)
    except FileNotFoundError:
        print(f"Error: The image file '{image_filename}' was not found in the script directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
