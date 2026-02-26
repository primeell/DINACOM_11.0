import json

with open("public/AI/model.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def check_model_config(obj):
    if isinstance(obj, dict):
        if obj.get("class_name") in ["Model", "Functional"] and "config" in obj:
            conf = obj["config"]
            print(f"Model named {conf.get('name')} has input_layers: {conf.get('input_layers') is not None}, output_layers: {conf.get('output_layers') is not None}")
            if "input_layers" in conf:
                print("input_layers:", conf["input_layers"])
            if "output_layers" in conf:
                print("output_layers:", conf["output_layers"])
        for k, v in obj.items():
            check_model_config(v)
    elif isinstance(obj, list):
        for x in obj:
            check_model_config(x)

check_model_config(data)
