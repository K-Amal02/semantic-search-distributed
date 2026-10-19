# src\encoders\joblib_encoder.py

from sentence_transformers import SentenceTransformer
from joblib import Parallel, delayed

# Charger le modèle IA
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Fonction de traitement parallèle
def encode_text(text):
    return model.encode(text)

# Fonction principale avec parallélisation
def encode_documents(docs, n_jobs=-1):
    """
    Encode une liste de documents en utilisant plusieurs cœurs CPU.
    :param docs: Liste de chaînes de caractères
    :param n_jobs: Nombre de jobs parallèles (-1 = tous les cœurs dispo)
    :return: Liste de vecteurs
    """
    embeddings = Parallel(n_jobs=n_jobs)(
        delayed(encode_text)(doc) for doc in docs
    )
    return embeddings
