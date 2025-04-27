import numpy as np
import os
from sentence_transformers import SentenceTransformer

class LabelGenerator:
    def __init__(paper_labels, label_path):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(paper_labels)
        np.save(os.path.join(label_path, 'paper_embeddings.npy'), embeddings)