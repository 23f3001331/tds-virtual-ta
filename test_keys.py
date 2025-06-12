import json

with open("discourse_posts.json", "r") as f:
    posts = json.load(f)

sample = posts[0]
print("Sample keys:", list(sample.keys()))
print("Sample content preview:", sample.get("cooked") or sample.get("raw") or "No content found.")
