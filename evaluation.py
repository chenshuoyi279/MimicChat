import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
# nltk.download('punkt')

def information_retrieval_evaluation(assistant_content, reference_text):
    # Combine assistant content into one text
    assistant_combined = ' '.join(assistant_content)

    # Create TF-IDF representation
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([assistant_combined, reference_text])

    # Calculate cosine similarity
    cos_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    return cos_similarity

def lexical_diversity_evaluation(assistant_content):
    # Tokenize the words in the assistant content
    words_flat = [word for content in assistant_content for word in word_tokenize(content.lower())]

    # Calculate Lexical Diversity
    lexical_diversity = len(set(words_flat)) / len(words_flat) if words_flat else 0

    return lexical_diversity

def syntactic_similarity_evaluation(assistant_content, reference_text):
    # Calculate average sentence length for assistant content
    assistant_sentences = [sent for content in assistant_content for sent in sent_tokenize(content)]
    avg_assistant_sentence_length = sum(len(word_tokenize(sent)) for sent in assistant_sentences) / len(assistant_sentences) if assistant_sentences else 0

    # Calculate average sentence length in the reference text
    reference_sentences = sent_tokenize(reference_text)
    avg_ref_sentence_length = sum(len(word_tokenize(sent)) for sent in reference_sentences) / len(reference_sentences) if reference_sentences else 0

    # Compare the averages (the closer they are, the more syntactically similar)
    syntactic_similarity = 1 - abs(avg_assistant_sentence_length - avg_ref_sentence_length) / max(avg_assistant_sentence_length, avg_ref_sentence_length)

    return syntactic_similarity

# Load from a JSON file
with open('conversation_data.json', 'r') as file:
    data = json.load(file)

conversation_history = data.get("conversation_history", [])
reference_text = data.get("reference_text", "")

# Extract only assistant content
assistant_content = [msg['content'] for msg in conversation_history if msg['role'] == 'assistant']

# Filter and combine reference text for specified person
person_name = "Joe"  # The person being mimicked
filtered_reference_text = []

for conversation in reference_text:
    for snippet in conversation:
        lines = snippet.split('\n')
        joe_lines = [line for line in lines if line.startswith(f"[Time: ") and f", Speaker: {person_name}]" in line]
        filtered_reference_text.extend(joe_lines)

combined_reference_text = ' '.join(filtered_reference_text)

# Information Retrieval Score
info_retrieval_score = information_retrieval_evaluation(assistant_content, combined_reference_text)

# Lexical Diversity Score
lexical_diversity_score = lexical_diversity_evaluation(assistant_content)

# Syntactic Similarity Score
syntactic_similarity_score = syntactic_similarity_evaluation(assistant_content, combined_reference_text)

print("Information Retrieval Score:", info_retrieval_score)
print("Lexical Diversity Score:", lexical_diversity_score)
print("Syntactic Similarity Score:", syntactic_similarity_score)
