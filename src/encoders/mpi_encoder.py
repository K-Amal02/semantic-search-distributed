# src\encoders\mpi_encoder.py

from mpi4py import MPI
from sentence_transformers import SentenceTransformer
from src.loader import load_documents_from_folder
import numpy as np
import time
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

if rank == 0:
    docs, _ = load_documents_from_folder("data/corpus_txt")
    chunks = [list(c) for c in np.array_split(docs, size)]
else:
    chunks = None

local_docs = comm.scatter(chunks, root=0)
start = time.time()
local_embeddings = model.encode(local_docs)
gathered = comm.gather(local_embeddings, root=0)

if rank == 0:
    full_embeddings = np.vstack(gathered)
    os.makedirs("embeddings", exist_ok=True)
    np.save("embeddings/embeddings.npy", full_embeddings)
    print(f"Temps d'encodage : {time.time() - start:.2f} s")
    print("Encodage distribue termine et sauvegarde.")  