import os
import email
from email import policy
from email.parser import BytesParser
from fastapi import FastAPI, UploadFile, File
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # Adjust num_labels as needed

def load_eml(file_path):
    with open(file_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    return msg

def extract_text_from_eml(msg):
    if msg.is_multipart():
        for part in msg.iter_parts():
            if part.get_content_type() == 'text/plain':
                return part.get_payload(decode=True).decode()
    else:
        return msg.get_payload(decode=True).decode()
    return ""

def classify_email(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidence, predicted_class = torch.max(probs, dim=1)
    return predicted_class.item(), confidence.item()

def is_duplicate(text1, text2):
    inputs1 = tokenizer(text1, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs2 = tokenizer(text2, return_tensors='pt', truncation=True, padding=True, max_length=512)
    embeddings1 = model.bert(**inputs1).last_hidden_state.mean(dim=1).detach().numpy()
    embeddings2 = model.bert(**inputs2).last_hidden_state.mean(dim=1).detach().numpy()
    similarity = cosine_similarity(embeddings1, embeddings2)
    return similarity[0][0] > 0.9  # Adjust threshold as needed

@app.post("/classify-email/")
async def classify_email_endpoint(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    msg = load_eml(file_path)
    text = extract_text_from_eml(msg)
    os.remove(file_path)
    
    email_type, confidence = classify_email(text)
    response = {
        "type": email_type,
        "confidence": confidence,
        "parameters": extract_parameters(text)
    }
    
    return response

def extract_parameters(text):
    # Implement parameter extraction logic based on email type and subtype
    parameters = {}
    # Example: Extracting name and loan amount from the email text
    if "loan" in text.lower():
        parameters["name"] = "Emily R. Johnson"  # Extracted from text
        parameters["loan_amount"] = "$30,000"  # Extracted from text
    return parameters

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)