import json

with open("public/AI/model.json", "r", encoding="utf-8") as f:
    data = json.load(f)

prefix = "sequential/"

stripped_count = 0
for group in data.get("weightsManifest", []):
    for w in group.get("weights", []):
        if w["name"].startswith(prefix):
            w["name"] = w["name"][len(prefix):]
            stripped_count += 1

with open("public/AI/model.json", "w", encoding="utf-8") as f:
    json.dump(data, f)
    
print(f"Stripped {prefix} from {stripped_count} weights.")
