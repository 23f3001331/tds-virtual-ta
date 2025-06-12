import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Paths
INPUT_FILE = "data/processed_discourse_posts.json"
INDEX_FILE = "data/faiss_index.idx"
META_FILE = "data/faiss_meta.pkl"

# Load processed data
with open(INPUT_FILE, "r") as f:
    topics = json.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract subthreads and metadata
subthreads = []
metadata = []

for topic in topics:
    posts = topic["posts"]
    topic_id = topic["topic_id"]
    topic_title = topic["topic_title"]

    # Grouping posts by their reply roots (basic subthread logic)
    subthread_texts = []
    current_text = f"Topic title: {topic_title}\n\n"
    for post in posts:
        current_text += post["content"] + "\n\n"
    subthreads.append(current_text.strip())
    metadata.append({
        "topic_id": topic_id,
        "topic_title": topic_title,
        "post_numbers": [p["post_number"] for p in posts],
        "urls": [p["url"] for p in posts]
    })

# Generate embeddings
print(f"Generating embeddings for {len(subthreads)} subthreads...")
embeddings = model.encode(subthreads, show_progress_bar=True)

# Create FAISS index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index and metadata
faiss.write_index(index, INDEX_FILE)
with open(META_FILE, "wb") as f:
    pickle.dump(metadata, f)

print(f"Saved FAISS index to {INDEX_FILE}")
print(f"Saved metadata to {META_FILE}")


