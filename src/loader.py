# src/loader.py

import os
import pymupdf  # PyMuPDF

def load_documents_from_folder(folder_path):
    documents = []
    filenames = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if filename.endswith(".txt"):
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
        elif filename.endswith(".pdf"):
            content = extract_text_from_pdf(path)
        else:
            continue
        documents.append(content)
        filenames.append(filename)
    return documents, filenames

def extract_text_from_pdf(path):
    text = ""
    doc = pymupdf.open(path)
    for page in doc:
        text += page.get_text()
    return text
