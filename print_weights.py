import json

with open("public/AI/model.json", "r", encoding="utf-8") as f:
    data = json.load(f)

actual_weights = []
for group in data.get("weightsManifest", []):
    for w in group.get("weights", []):
        actual_weights.append(w["name"])

print(f"Sample weights:")
for w in actual_weights[:20]:
    print(w)
