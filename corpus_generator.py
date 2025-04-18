from sentence_transformers import SentenceTransformer
import numpy as np
import os
import re 
import glob
import tarfile 
import gzip 
import pandas as pd

class CorpusGenerator:
    def __init__(self, dataset_root, embedding_dir):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dataset_root = dataset_root
        self.embedding_dir = embedding_dir

    def generate_labels(self):
        paper_labels = ['reinforcement learning'] # TODO: Use LLM call here
        embeddings = self.model.encode(paper_labels)
        np.save(os.path.join(self.embedding_dir, 'label_embeddings.npy'), embeddings)

    # Define a helper function to extract title and abstract from LaTeX content.
    def extract_title_and_abstract(self, latex_content):
        """
        Extracts title and abstract from LaTeX content.

        Assumes:
        - Title is defined with \title{...}
        - Abstract is within \begin{abstract} ... \end{abstract}
        """
        title_match = re.search(r'\\title\s*\{(.+?)\}', latex_content, re.DOTALL)
        title = title_match.group(1).strip() if title_match else None

        abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', latex_content, re.DOTALL)
        abstract = abstract_match.group(1).strip() if abstract_match else None

        return title, abstract
    
    def extract_files(self):
        # Process each .gz file: open using tarfile (r:gz mode) and extract the first .tex file.
        paper_data = []  # List to hold dictionaries with extracted data
        files = []
        for ext in ['.gz', '.zip']:
            files.extend(glob.glob(os.path.join(self.dataset_root, '*' + 'ext')))

        print(files)
        for file in files:
            drive_info = os.path.basename(file)
            content = ""
            try:
                try:
                    # Attempt to open as a tar archive.
                    with tarfile.open(file, "r:gz") as tar:
                        # Find all members ending with '.tex'
                        tex_members = [member for member in tar.getmembers() if member.name.endswith('.tex')]
                        if tex_members:
                            member = tex_members[0]
                            # FIXME: This only takes one tex file as a representative of the entire paper
                            # For papers that have multiple, what if we pick a bad representative?
                            f = tar.extractfile(member)
                            if f is not None:
                                content = f.read().decode('utf-8', errors='ignore')
                            else:
                                raise Exception("Failed to extract .tex file from tar archive.")
                        else:
                            raise Exception("No .tex file found in tar archive.")
                except (tarfile.ReadError, Exception) as e:
                    # If reading as a tar archive fails, fall back to plain gzip.
                    print(f"Tar archive read error for {file}: {e}. Trying to open as a plain gzip file.")
                    with gzip.open(file, "rt", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                # Extract title and abstract from the content.
                title, abstract = self.extract_title_and_abstract(content)

                paper_data.append({
                    'title': title,
                    'abstract': abstract,
                    'drive_file': drive_info
                })
            except Exception as e:
                print(f"Error processing {file}: {e}")
                paper_data.append({
                    'title': "Error reading file",
                    'abstract': "Error reading file",
                    'drive_file': drive_info
                })

        print(f"Extracted data from {len(paper_data)} files.")

        # Create a DataFrame from the extracted data.
        # TODO: Use pickle or something to save this to disk
        papers_df = pd.DataFrame(paper_data)
        print(papers_df.head())
