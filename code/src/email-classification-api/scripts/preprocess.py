import os
import re
from email import policy
from email.parser import BytesParser
from sklearn.feature_extraction.text import CountVectorizer

def extract_email_content(eml_file_path):
    with open(eml_file_path, 'rb') as file:
        email = BytesParser(policy=policy.default).parse(file)
    # Extracting the email body (ignoring headers)
    return email.get_body(preferencelist=('plain')).get_content()

def preprocess_email_content(content):
    # Remove special characters and convert to lowercase
    content = re.sub(r"[^a-zA-Z\s]", "", content)
    content = content.lower()
    return content

def get_features(data):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data)
    return X, vectorizer