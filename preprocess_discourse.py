import json
from tqdm import tqdm
from pathlib import Path

INPUT_FILE = "discourse_posts.json"
OUTPUT_FILE = "data/processed_discourse_posts.json"

# Ensure the data folder exists
Path("data").mkdir(parents=True, exist_ok=True)

# Load the raw Discourse posts
with open(INPUT_FILE, "r") as f:
    posts = json.load(f)

# Group posts by topic and subthreads
topics = {}
for post in tqdm(posts, desc="Organizing posts by topic"):
    topic_id = post["topic_id"]
    topic_title = post["topic_title"]
    post_number = post["post_number"]
    reply_to = post["reply_to_post_number"]
    content = post.get("content", "").strip()

    if topic_id not in topics:
        topics[topic_id] = {
            "topic_id": topic_id,
            "topic_title": topic_title,
            "posts": []
        }
    
    topics[topic_id]["posts"].append({
        "post_number": post_number,
        "reply_to": reply_to,
        "content": content,
        "url": post["url"]
    })

# Save the cleaned and grouped data
with open(OUTPUT_FILE, "w") as f:
    json.dump(list(topics.values()), f, indent=2)

print(f"Processed and saved {len(topics)} topics to {OUTPUT_FILE}")


