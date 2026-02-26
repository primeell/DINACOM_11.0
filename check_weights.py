import json

with open("public/AI/model.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Collect all expected weights from topology
expected_layers = set()
def collect_layers(obj):
    if isinstance(obj, dict):
        if obj.get("class_name") and "config" in obj and "name" in obj["config"]:
            expected_layers.add(obj["config"]["name"])
        for k, v in obj.items():
            collect_layers(v)
    elif isinstance(obj, list):
        for x in obj:
            collect_layers(x)

collect_layers(data.get("modelTopology", {}))
print(f"Total expected layer names: {len(expected_layers)}")
sample_expected = list(expected_layers)[:10]

# Collect all actual weights from manifest
actual_weights = []
for group in data.get("weightsManifest", []):
    for w in group.get("weights", []):
        actual_weights.append(w["name"])

print(f"Total weight names in manifest: {len(actual_weights)}")
sample_actual = actual_weights[:10]

print("\nSample expected layers:")
print(sample_expected)

print("\nSample actual weights:")
print(sample_actual)
