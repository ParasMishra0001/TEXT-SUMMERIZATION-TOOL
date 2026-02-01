import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import heapq

nltk.download('punkt')

def summarize_text(text, num_sentences=3):
    sentences = sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return text

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)

    sentence_scores = {}
    for i in range(len(sentences)):
        sentence_scores[i] = tfidf_matrix[i].sum()

    top_sentences = heapq.nlargest(
        num_sentences, sentence_scores, key=sentence_scores.get
    )
    top_sentences.sort()

    summary = " ".join([sentences[i] for i in top_sentences])
    return summary


# ===== USER INPUT =====
print("enter the text (press the enter for end):")
user_text = input()

summary = summarize_text(user_text, num_sentences=2)

print("\nSummary:")
print(summary)
