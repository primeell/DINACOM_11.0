import json

try:
    with open("model.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Unwrap model_config
    if "modelTopology" in data and "model_config" in data["modelTopology"]:
        mc = data["modelTopology"]["model_config"]
        data["modelTopology"]["class_name"] = mc["class_name"]
        data["modelTopology"]["config"] = mc["config"]
        data["modelTopology"]["keras_version"] = "2.15.0"
        del data["modelTopology"]["model_config"]

    # 2. Recursive replace batch_shape with batch_input_shape and Functional with Model, remove module/registered_name
    def clean(obj):
        if isinstance(obj, list):
            return [clean(x) for x in obj]
        elif isinstance(obj, dict):
            if obj.get("class_name") == "DTypePolicy" and "config" in obj and "name" in obj["config"]:
                return obj["config"]["name"]
            
            if obj.get("class_name") == "InputLayer" and "config" in obj and "batch_shape" in obj["config"]:
                obj["config"]["batch_input_shape"] = obj["config"]["batch_shape"]
                del obj["config"]["batch_shape"]
                
            if obj.get("class_name") == "Functional":
                obj["class_name"] = "Model"
                
            for k in ["module", "registered_name", "build_config", "compile_config"]:
                if k in obj:
                    del obj[k]
                    
            for k, v in obj.items():
                obj[k] = clean(v)
                
        return obj

    data = clean(data)
    data["format"] = "layers-model"

    with open("public/AI/model.json", "w", encoding="utf-8") as f:
        json.dump(data, f)
        
    print("Cleaned!")
except Exception as e:
    print("Error:", e)
