from recommender import ResearchPaperRecommender
from corpus_generator import *
from IPython.display import display
import sys 
import os 

def generate_corpus(embedding_dir):
    DATASET_ROOT = './dataset'
    c = CorpusGenerator(DATASET_ROOT, embedding_dir)
    # TODO: Ensure that we save both the paper and label embeddings
    c.extract_files()
    c.generate_labels()

if __name__ == "__main__":
    # Provide paths to dataset, paper labels, and embedding vectors
    # The latter two should hopefully be dictionaries of title, vector pairs
    EMBEDDING_DIR = './embeddings'
    PAPER_EMBEDDINGS = os.path.join(EMBEDDING_DIR, 'paper_embeddings.npy')
    LABEL_EMBEDDINGS = os.path.join(EMBEDDING_DIR, 'label_embeddings.npy')
    if not os.path.exists(PAPER_EMBEDDINGS) or not os.path.exists(LABEL_EMBEDDINGS):
        generate_corpus(EMBEDDING_DIR)
    try:
        rec = ResearchPaperRecommender(LABEL_EMBEDDINGS, PAPER_EMBEDDINGS)
        print("Research Paper Recommender System initialized successfully!")
        print("Type 'exit' to quit the program.\n")
    except Exception as e:
        print(f"Failed to initialize recommender: {str(e)}")
        sys.exit()

    # Interactive loop
    while True:
        try:
            # Get user input
            query = input("\nEnter your research interest or topic (or 'exit' to quit): ").strip()

            if query.lower() == 'exit':
                print("Exiting recommender system...")
                break

            if not query:
                print("Please enter a valid query.")
                continue

            # Get recommendations
            print(f"\nSearching for papers related to: '{query}'...")
            results = rec.recommend(query)

            # Display results
            if not results.empty:
                print("\nTop Recommendations:")
                display(results[['title', 'labels']])

                # Show details option
                show_details = input("\nWould you like to see abstracts? (y/n): ").lower()
                if show_details == 'y':
                    display(results[['title', 'abstract', 'url']])
            else:
                print("No relevant papers found. Try a different query.")

        except KeyboardInterrupt:
            print("\nExiting recommender system...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            continue