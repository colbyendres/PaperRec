import os
import re
import time
import sys 
import requests 
import json 

# The recommendation engine gave me some papers written in Thai
# Use this kludge to filter out papers not written in English
# (or at least anything not using a subset of the Latin alphabet)
IS_PROBABLY_ENGLISH = re.compile(r'[a-zA-Z]+')

# TODO: Use something less dumb
def is_english_title(title):
    return bool(re.match(IS_PROBABLY_ENGLISH, title))

# Semantic Scholar is rather stringent with its rate limits if you don't have an API key
# Apply exponential backoff for HTTP requests to play nice
def get_with_backoff(url, params, attempts=5):
    backoff = 5
    for _ in range(attempts):
        r = requests.get(url, params=params)
        if r.status_code == 429:
            backoff *= 2
            time.sleep(backoff)
            continue
        return r.json()

    # Failed to get data
    return None 

# Retrieve relevant papers from Semantic Scholar for given query
def search_semantic_scholar(query, paper_limit=100, attempts=5):
    url = "http://api.semanticscholar.org/graph/v1/paper/search/"
    params = {
        "query": query,
        "limit": paper_limit,
        "fields": "title,paperId",
        "fieldsOfStudy" : "Mathematics", # NOTE: This can be removed, if needed
        "sort" : "citationCount:desc"
    }

    rsp = get_with_backoff(url, params)
    return rsp.get('data', [])

# Returns top k recommendations for a given paper
def find_recommendations(ss_id, limit):
    url = f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{ss_id}"
    params = {
        'fields': 'title,paperId', 
        'limit': limit, 
        'sort' : 'citationCount:desc' # Sort by highest citation count
        }
    rsp = get_with_backoff(url, params)
    return rsp.get('recommendedPapers', [])

# Write the papers to a JSON file
def write_json(queries, papers_per_query, file_name, recs_per_paper):
    data = {}
    # Multiplicative factor to which oversample recommendations
    # This is necessary, as some recommendations may be ineligible
    OVERSAMPLE = 2
    for q in queries:
        papers = search_semantic_scholar(q, papers_per_query)
        if not papers:
            print(f"Failed to retrieve query {q}")
            continue

        # Track papers for the JSON
        papers_with_recs = []
        rec_papers = 0
        for paper in papers:
            recs = find_recommendations(paper['paperId'], OVERSAMPLE * recs_per_paper)
            
            # Reject any paper that isn't written in English
            # Trim list to meet limit, if not too many disqualifications
            recs = [rec for rec in recs if is_english_title(rec.get('title', ''))][:recs_per_paper]

            # Reject any paper that has no English recommendations
            if not recs:
                continue 

            papers_with_recs.append(paper)
            rec_papers += 1
            paper['recommendations'] = recs
            if rec_papers == recs_per_paper: # Hit recommendation target, can stop now
                break 
            
        data[q] = papers_with_recs 

    with open(file_name, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    num_papers = 100
    recs_per_paper = 10
    if len(sys.argv) > 2:
        num_papers = int(sys.argv[1])
        recs_per_paper = int(sys.argv[2])

    QUERIES = ['machine learning', 'information retrieval', 'mathematical physics', 'commutative algebra']
    FILE_PATH = '../papers_large.json'
    
    # A query can return at max 100 results (API limitation)
    papers_per_query = num_papers // len(QUERIES)
    assert papers_per_query <= 100 
    
    write_json(QUERIES, papers_per_query, FILE_PATH, recs_per_paper)