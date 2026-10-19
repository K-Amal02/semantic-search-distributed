# src\encoders\multiprocessing_encoder.py

from multiprocessing import Pool, cpu_count
from sentence_transformers import SentenceTransformer

import numpy as np

def encode_chunk(docs):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return model.encode(docs)

def parallel_encode(docs, num_processes=None):
    if num_processes is None:
        num_processes = min(cpu_count(), len(docs))
    chunks = [list(c) for c in np.array_split(docs, num_processes)]
    with Pool(num_processes) as pool:
        result = pool.map(encode_chunk, chunks)
    return np.vstack(result)
