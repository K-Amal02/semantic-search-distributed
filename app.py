import streamlit as st
from src.loader import load_documents_from_folder
from src.search_engine import encode_documents, search, save_results

st.title("🔍 Moteur de Recherche Sémantique Distribué")

@st.cache_data(show_spinner="Encodage des documents...")

# Chargement initial des documents

def charger_et_encoder():
    docs, filenames = load_documents_from_folder("data/corpus_txt")
    embeddings = encode_documents(docs)
    return docs, filenames, embeddings

docs, filenames, embeddings = charger_et_encoder()
st.write(f"{len(docs)} documents chargés.")

# Encodage sémantique
with st.spinner("Encodage des documents..."):
    embeddings = encode_documents(docs)

# Interface utilisateur
query = st.text_input("Entrez votre requête :")
if query:
    with st.spinner("Recherche en cours..."):
        results = search(query, docs, embeddings)
        for i, (text, score) in enumerate(results):
            st.write(f"### Résultat {i+1} (Score : {score:.2f})")
            st.write(text[:500] + "...")
    
    # Sauvegarder les résultats
    save_results(query, results)
    st.write("Résultats enregistrés dans 'search_results.txt'.")
