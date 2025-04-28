import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

class ResearchPaperRecommender:
    def __init__(self, df_path):
        self.df = pd.read_csv(df_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.label_embeddings = self._calculate_label_embeddings(zip(self.df['title'], self.df['labels']))
        self.title_embeddings = self._calculate_title_embeddings(self.df['title'])

    # Precompute embeddings of each label
    # NOTE: This can be made redundant if we store the embeddings instead of the plaintext labels
    def _calculate_label_embeddings(self, data):
        embeddings = {}
        for title, labels in data:
            s = self.model.encode(labels)
            embeddings[title] = s / np.linalg.norm(s)
        return embeddings
    
    def _calculate_title_embeddings(self, titles):
        embeddings = {}
        for title in titles:
            s = self.model.encode(title)
            embeddings[title] = s / np.linalg.norm(s)
        return embeddings
    
    # Find paper in dataframe closest to query_title
    # Used as a fallback in case the user asks for a paper we don't know about
    def find_closest_title(self, query_title):
        query_embedding = self.model.encode(query_title)
        query_norm = np.linalg.norm(query_embedding)
        best_score = 0
        best_title = ''

        for title, embedding in self.title_embeddings.items():
            s = np.dot(query_embedding, embedding) / query_norm
            if s > best_score:
                best_score = s 
                best_title = title
        
        return best_title

    def recommend(self, title, top_n=5):
        try:
            query_row = self.df[self.df['title'] == title].iloc[0]
        except IndexError:
            best_title = self.find_closest_title(title)
            print(f"LOG: Title not found in database, using closest paper {best_title}")
            return self.recommend(best_title, top_n)

        # Narrow our search space to only include papers in the same cluster as the query
        centroid_paper = query_row['cluster']
        similar_cluster_df = self.df[(self.df['cluster'] == centroid_paper) & (self.df['title'] != title)].copy()

        if similar_cluster_df.empty:
            print(f"No other papers found in the same cluster as '{title}'.")
            return []

        # Compute similarity scores between query and candidate embeddings
        query_embedding = self.label_embeddings[title]
        candidate_titles = list(similar_cluster_df['title'])
        similarity_scores = np.array([np.dot(query_embedding, self.label_embeddings[cand]) for cand in candidate_titles])
        best_idx = np.argsort(similarity_scores)[-top_n:]

        return [candidate_titles[i] for i in best_idx]
        
    
if __name__ == '__main__':
    DATAFRAME_PATH = './dataset/test.csv'
    r = ResearchPaperRecommender(DATAFRAME_PATH)
    while True:
        title = input('Input a title for similar papers (q to quit): ')
        if title == 'q':
            break
        results = r.recommend(title)
        for i, res in enumerate(results):
            print(f"{i+1}: {res}")


    