import os
import json
import re

TDS_CONTENT_DIR = "markdown_files"
OUTPUT_JSON = "data/processed_tds_content.json"

def extract_front_matter_and_content(md_text):
    front_matter = {}
    content_lines = []

    in_front_matter = False
    for line in md_text.splitlines():
        if line.strip() == "---":
            in_front_matter = not in_front_matter
            continue
        if in_front_matter:
            if ":" in line:
                key, value = line.split(":", 1)
                front_matter[key.strip()] = value.strip().strip('"')
        else:
            content_lines.append(line)

    content = "\n".join(content_lines).strip()
    return front_matter, content

def chunk_markdown_content(text, max_length=500):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

all_chunks = []

for filename in os.listdir(TDS_CONTENT_DIR):
    if filename.endswith(".md"):
        filepath = os.path.join(TDS_CONTENT_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            md_text = f.read()

        metadata, content = extract_front_matter_and_content(md_text)
        if not content:
            continue

        chunks = chunk_markdown_content(content)
        for chunk in chunks:
            all_chunks.append({
                "content": chunk,
                "title": metadata.get("title", ""),
                "url": metadata.get("original_url", ""),
                "downloaded_at": metadata.get("downloaded_at", "")
            })

# Ensure 'data/' directory exists
os.makedirs("data", exist_ok=True)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2)

print(f"✅ Saved {len(all_chunks)} chunks to {OUTPUT_JSON}")

