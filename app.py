from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json

app = Flask(__name__)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS index
index = faiss.read_index("data/tds_index.faiss")

# Load metadata
with open("data/faiss_meta.pkl", "rb") as f:
    metadata = pickle.load(f)

# Load chunked content
with open("data/tds_all_chunks.json", "r") as f:
    all_chunks = json.load(f)

@app.route("/answer", methods=["POST"])
def answer():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Missing 'question' in request"}), 400

    # Get embedding for the question
    question_embedding = model.encode([question])

    # Search FAISS index for top 3 most relevant chunks
    D, I = index.search(question_embedding, k=3)

    # Collect answers
    results = []
    for idx in I[0]:
        chunk_text = all_chunks[idx]
        meta = metadata[idx]
        source = meta.get("source", "")
        title = meta.get("title", "TDS Content" if "tds" in source.lower() else "Discourse Post")

        results.append({
            "answer": chunk_text.strip(),
            "source": source,
            "title": title
        })

    if not results:
        return jsonify({"answer": "Sorry, no relevant answer found.", "source": "", "suggestions": []})

    # Main answer
    main = results[0]
    suggestions = [
        {
            "text": suggestion["answer"][:200] + "...",
            "url": suggestion["source"]
        }
        for suggestion in results[1:]
    ]

    return jsonify({
        "answer": main["answer"],
        "source": main["source"],
        "suggestions": suggestions
    })

if __name__ == "__main__":
    app.run(debug=True,port=5050)




