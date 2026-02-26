import json

with open("public/AI/model.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for group in data.get("weightsManifest", []):
    for w in group.get("weights", []):
        name = w["name"]
        
        # 1. Strip 'sequential/' if it's there
        if name.startswith("sequential/"):
            name = name[len("sequential/"):]
            
        # 2. Check if it belongs to dense or dense_1
        if name.startswith("dense/") or name.startswith("dense_1/"):
            w["name"] = name
        else:
            # It belongs to mobilenetv2_1.00_224
            # Only prepend if it doesn't already have it
            if not name.startswith("mobilenetv2_1.00_224/"):
                w["name"] = "mobilenetv2_1.00_224/" + name
            else:
                w["name"] = name

with open("public/AI/model.json", "w", encoding="utf-8") as f:
    json.dump(data, f)

print("Rewrote weightsManifest successfully!")
