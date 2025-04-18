import numpy as np
import re
import os
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

class ResearchPaperRecommender:
    def __init__(self, label_path, paper_path):
        # Load embeddingsa
        self.paper_embeddings = np.load(paper_path).astype('float32')
        num_papers = len(self.paper_embeddings)

        # Create synthetic paper data (using first 'num_papers' labels)
        self.paper_labels = np.load(label_path).astype('float32')

        # Create dataframe
        self.papers_df = pd.DataFrame({
            'title': [f"Paper {i+1}" for i in range(num_papers)],
            'labels': self.paper_labels,
            'abstract': [f"This paper discusses {label.lower()}. It presents novel findings." for label in self.paper_labels],
            'url': [f"http://example.com/paper_{i+1}" for i in range(num_papers)]
        })

        # Initialize models
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self._prepare_indices()
        self._prepare_bm25()

    def _prepare_indices(self):
        """Prepare FAISS index for semantic search"""
        self.dimension = self.paper_embeddings.shape[1]
        self.semantic_index = faiss.IndexFlatIP(self.dimension)
        faiss.normalize_L2(self.paper_embeddings)
        self.semantic_index.add(self.paper_embeddings)

    def _prepare_bm25(self):
        """Prepare BM25 index for keyword search"""
        processed_labels = self.papers_df['labels'].apply(self._preprocess_text)
        self.tokenized_labels = [word_tokenize(doc) for doc in processed_labels]
        self.bm25 = BM25Okapi(self.tokenized_labels)

    def _preprocess_text(self, text):
        """Basic text cleaning"""
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        return text

    def recommend(self, query, top_k=5, hybrid=True):
        """Get paper recommendations"""
        try:
            # Preprocess query
            processed_query = self._preprocess_text(query)

            # Semantic search
            query_embedding = self.embedding_model.encode([query], convert_to_tensor=False).astype('float32')
            faiss.normalize_L2(query_embedding)
            D, I = self.semantic_index.search(query_embedding, top_k*2)  # D = distances, I = indices

            # Keyword search
            tokenized_query = word_tokenize(processed_query)
            bm25_scores = self.bm25.get_scores(tokenized_query)

            if hybrid:
                # Get top BM25 scores for the same papers
                bm25_top_scores = np.array([bm25_scores[i] for i in I[0]])

                # Normalize scores
                semantic_norm = D[0] / np.max(D[0])
                bm25_norm = bm25_top_scores / np.max(bm25_top_scores)

                # Combine scores
                combined_scores = 0.7 * semantic_norm + 0.3 * bm25_norm
                top_indices = I[0][np.argsort(combined_scores)[::-1][:top_k]]
            else:
                top_indices = I[0][:top_k]

            return self.papers_df.iloc[top_indices]
        except Exception as e:
            print(f"Error during recommendation: {str(e)}")
            return pd.DataFrame()

