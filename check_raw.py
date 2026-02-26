import json

with open("model.json", "r", encoding="utf-8") as f:
    orig = json.load(f)

print("Original raw weights:")
for group in orig.get("weightsManifest", []):
    for w in group.get("weights", [])[:10]:
        print(w["name"])

print("\nNested layers in topology:")
mc = orig.get("modelTopology", {}).get("model_config", {}).get("config", {})
if "layers" in mc:
    for l in mc["layers"]:
        print(f"Layer: {l.get('class_name')} - {l.get('config', {}).get('name')}")
