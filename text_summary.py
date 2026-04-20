import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import PyPDF2
from heapq import nlargest
import re

stopwords = list(STOP_WORDS)
nlp = spacy.load('en_core_web_sm')

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    text = re.sub(r'\n+', ' ', text)  # remove new lines
    text = re.sub(r'[^\w\s.,]', '', text)  # remove unwanted symbols
    return text

def read_text_from_file(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()
    print("Text extracted from PDF:", text)
    return text

def summary_text(text):
    text = clean_text(text)
    doc = nlp(text)

    word_freq = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            word_freq[word.text] = word_freq.get(word.text, 0) + 1

    if not word_freq:
        return "", doc, len(text.split()), 0

    max_freq = max(word_freq.values())
    for word in word_freq:
        word_freq[word] /= max_freq

    def is_valid_sentence(sent):
        if len(sent.text) < 20:
            return False
        if '=' in sent.text or ';' in sent.text:
            return False
        return True

    sent_token = [sent for sent in doc.sents if is_valid_sentence(sent)]

    sent_score = {}
    for sent in sent_token:
        for word in sent:
            if word.text in word_freq:
                sent_score[sent] = sent_score.get(sent, 0) + word_freq[word.text]

    select_len = max(3, int(len(sent_token) * 0.25))

    summary_sentences = nlargest(select_len, sent_score, key=sent_score.get)

    summary = '\n\n'.join([sent.text.strip() for sent in summary_sentences])

    return summary, doc, len(text.split()), len(summary.split())
if __name__ == "__main__":
    # Test read_text_from_file function
    filename = "textfile.pdf"  # Replace "your_pdf_file.pdf" with the path to your PDF file
    text = read_text_from_file(filename)
    print("Text extracted from PDF:", text)

    # Test summary_text function
    summary, doc, len_of_text, len_of_summary = summary_text(text)
    print("Summary:", summary)
    print("Original Text:", doc)
    print("Length of Text:", len_of_text)
    print("Length of Summary:", len_of_summary)
