import json
import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss

# Load Discourse JSON
with open("data/processed_discourse_posts.json", "r") as f:
    discourse_data = json.load(f)

# Load TDS JSON
with open("data/processed_tds_content.json", "r") as f:
    tds_data = json.load(f)

# Extract chunks and metadata
texts = []
metadata = []

# Discourse
for thread in discourse_data:
    for post in thread["posts"]:
        content = post.get("content", "").strip()
        if content:
            texts.append(content)
            metadata.append({
                "source": post.get("url", ""),
                "title": thread.get("topic_title", "Discourse Post")
            })

# TDS
for item in tds_data:
    content = item.get("content", "").strip()
    if content:
        texts.append(content)
        metadata.append({
            "source": item.get("url", ""),
            "title": item.get("title", "TDS Content")
        })

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index
os.makedirs("data", exist_ok=True)
faiss.write_index(index, "data/tds_index.faiss")

# Save metadata
with open("data/faiss_meta.pkl", "wb") as f:
    pickle.dump(metadata, f)

# Save text chunks
with open("data/tds_all_chunks.json", "w") as f:
    json.dump(texts, f, indent=2)

print("✅ Index and all data saved successfully!")



