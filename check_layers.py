import json

with open("public/AI/model.json", "r", encoding="utf-8") as f:
    data = json.load(f)

layers = data.get("modelTopology", {}).get("config", {}).get("layers", [])
if not layers:
    print("No layers found in sequential config, checking inner model")
    # Might be wrapped
    layers = data.get("modelTopology", {}).get("model_config", {}).get("config", {}).get("layers", [])

print("Last 3 layers:")
for l in layers[-3:]:
    print(f"{l.get('class_name')} - {l.get('config', {}).get('activation')}")

