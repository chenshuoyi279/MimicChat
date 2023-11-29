# analyze the result from the similarity search each time augmenting a new prompt
# return: Emotion Analysis, Contextual Analysis

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

def update_emotion_scores(conversation, previous_scores):
    # Initialize VADER sentiment analyzer
    sentiment_analyzer = SentimentIntensityAnalyzer()

    # Split the conversation into lines
    lines = conversation.split('\n')

    # Check if previous_scores is not None and is a dictionary
    if previous_scores is None:
        previous_scores = {}
    elif not isinstance(previous_scores, dict):
        raise ValueError("previous_scores must be a dictionary")

    # Process each line in the conversation
    for line in lines:
        # Extract speaker and text using regular expression
        match = re.search(r'\[Time: [^]]*\, Speaker: (\w+)\] (.+)', line)
        if match:
            speaker, text = match.groups()
            # Perform emotion analysis
            scores = sentiment_analyzer.polarity_scores(text)
            # Update cumulative scores
            if speaker in previous_scores:
                previous_scores[speaker] += scores['compound']
            else:
                previous_scores[speaker] = scores['compound']

    return previous_scores

# Example usage
conversation = """
[Time: 00:02:38, Speaker: Joe] They scraped the fucking like.
[Time: 00:02:39, Speaker: Frank] My mom told me that.
...
"""  # Include the entire conversation string

previous_scores = {"Joe": 0.5, "Frank": 0.5}  # Example previous scores
updated_scores = update_emotion_scores(conversation, previous_scores)
print(updated_scores)
