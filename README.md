# PaperRec
This is a paper recommendation algorithm, designed as a course project for CSCE 670. The recommendation process is a two-stage pipeline and is based on the idea of paper labels. We first assign all papers to $k$ clusters, where $k$ is a tuned hyperparameter. Through the use of a LLM, we generate specific labels for each paper, which will be the backbone of our recommendation engine. For a given query title, we first find the constituent cluster. Then, using similarity scores between embeddings of the intracluster labels, we give our recommendation for related papers.

# Build Instructions
We have already curated a dataset with all of the relevant info, which can be found here. If you want further details on the preprocessing and clustering stage, see the contained Jupyter notebook. Building is simple:
```
git clone git@github.com:colbyendres/PaperRec.git
pip install -r requirements.txt
python recommender.py
```
Download the embeddings pkl file https://drive.google.com/file/d/1Blyt4V5qiPfGUNPzat8ok0LzDFEn3R2n/view?usp=sharing

# Custom Data
If you would like to supply your own dataset, the recommendation engine requires a DataFrame with the following fields:
```
title: Paper Title
labels: A space-separated string of labels pertinent to the given paper
cluster: The centroid title of the cluster containing 'title'. This must be a title present in the dataframe
```
With this, you can use your custom data by modifying the `DATASET_FILE` variable within `recommender.py`


