import json

with open("public/AI/model.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def is_keras_tensor(obj):
    return isinstance(obj, dict) and obj.get("class_name") == "__keras_tensor__" and "config" in obj and "keras_history" in obj["config"]

def convert_tensor(obj):
    # Returns [layer_name, node_index, tensor_index, {}]
    history = obj["config"]["keras_history"]
    return [history[0], history[1], history[2], {}]

def fix_inbound_nodes(nodes):
    new_nodes = []
    for node in nodes:
        # node in Keras 3 is usually a dict: {"args": [...], "kwargs": {...}}
        if isinstance(node, dict):
            args = node.get("args", [])
            flat_args = []
            for arg in args:
                if isinstance(arg, list):
                    # Lists of tensors, like for Add or Concatenate
                    nested = [convert_tensor(x) for x in arg if is_keras_tensor(x)]
                    if nested:
                        flat_args.extend(nested) # in Keras 2, Add's inbound node is [[ ["l1",0,0,{}], ["l2",0,0,{}] ]]
                elif is_keras_tensor(arg):
                    flat_args.append(convert_tensor(arg))
            
            if flat_args:
                new_nodes.append(flat_args)
        else:
            # Maybe already Keras 2 format?
            new_nodes.append(node)
    return new_nodes

def fix_all(obj):
    if isinstance(obj, dict):
        if "inbound_nodes" in obj and isinstance(obj["inbound_nodes"], list):
            obj["inbound_nodes"] = fix_inbound_nodes(obj["inbound_nodes"])
            
        for k, v in obj.items():
            fix_all(v)
    elif isinstance(obj, list):
        for x in obj:
            fix_all(x)

# First run the general python_fix cleanings
def general_clean(obj):
    if isinstance(obj, list):
        return [general_clean(x) for x in obj]
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
            obj[k] = general_clean(v)
    return obj

# Unwrap top-level model_config if needed (handled strictly)
if "modelTopology" in data and "model_config" in data["modelTopology"]:
    mc = data["modelTopology"]["model_config"]
    if "class_name" in mc:
        data["modelTopology"]["class_name"] = mc["class_name"]
        data["modelTopology"]["config"] = mc["config"]
        data["modelTopology"]["keras_version"] = "2.15.0"
        del data["modelTopology"]["model_config"]

# We have the raw model.json again from earlier back file, but actually `public/AI/model.json`
# has already gone through python_fix... Wait!
# The original model.json has `__keras_tensor__` with kwargs {"mask": null} sometimes. But `args` has all the inputs.
# I just need to load the original `model.json` from the root directory to guarantee we do this from scratch.
with open("model.json", "r", encoding="utf-8") as f:
    orig_data = json.load(f)

# 1. Unwrap
if "modelTopology" in orig_data and "model_config" in orig_data["modelTopology"]:
    mc = orig_data["modelTopology"]["model_config"]
    orig_data["modelTopology"]["class_name"] = mc["class_name"]
    orig_data["modelTopology"]["config"] = mc["config"]
    orig_data["modelTopology"]["keras_version"] = "2.15.0"
    del orig_data["modelTopology"]["model_config"]

# 2. Fix Inbound Nodes FIRST (so we extract tensors from dicts)
fix_all(orig_data)

# 3. Apply general clean
orig_data = general_clean(orig_data)
orig_data["format"] = "layers-model"

with open("public/AI/model.json", "w", encoding="utf-8") as f:
    json.dump(orig_data, f, indent=2)

print("Succesfully rewritten deeply.")
