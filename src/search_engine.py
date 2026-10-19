from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def encode_documents(docs):
    return model.encode(docs, show_progress_bar=True)

def search(query, docs, embeddings, top_k=5):
    query_embedding = model.encode([query], show_progress_bar=False)
    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = [(docs[i], scores[i]) for i in top_indices]
    return results

def save_results(query, results, filename="search_results.txt"):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Résultats pour la requête : {query}\n\n")
        for i, (doc, score) in enumerate(results):
            f.write(f"--- Résultat {i+1} (score={score:.2f}) ---\n")
            f.write(doc[:500] + "\n\n")
