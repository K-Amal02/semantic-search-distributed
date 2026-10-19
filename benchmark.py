# benchmark.py
# Compare les stratégies d'encodage : séquentiel, joblib, multiprocessing.
# (MPI se lance à part : mpiexec -n 2 python -m src.encoders.mpi_encoder)

import time


def chronometrer(nom, fonction, docs):
    debut = time.time()
    fonction(docs)
    duree = time.time() - debut
    print(f"✅ {nom} : {duree:.2f} s")
    return duree


def main():
    # Imports dans main() : sous Windows, chaque processus enfant ré-importe
    # ce fichier, et un import en haut rechargerait le modèle inutilement.
    from sentence_transformers import SentenceTransformer
    from src.loader import load_documents_from_folder
    from src.encoders.joblib_encoder import encode_documents as encode_joblib
    from src.encoders.multiprocessing_encoder import parallel_encode

    print("📄 Chargement des documents...")
    docs, _ = load_documents_from_folder("data/corpus_txt")
    print(f"{len(docs)} documents chargés.\n")

    # Séquentiel : le modèle est chargé avant le chronométrage
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    resultats = {
        "Séquentiel": chronometrer("Séquentiel", model.encode, docs),
        "joblib": chronometrer("joblib", encode_joblib, docs),
        "multiprocessing": chronometrer("multiprocessing", parallel_encode, docs),
    }

    print("\n📊 Résumé")
    for nom, duree in resultats.items():
        print(f"  {nom:<18} {duree:>8.2f} s")


if __name__ == "__main__":
    main()