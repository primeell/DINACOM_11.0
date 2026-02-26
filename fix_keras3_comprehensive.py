import json

with open("model.json", "r", encoding="utf-8") as f:
    orig_data = json.load(f)

depthwise_layers = set()
def collect_depthwise(obj):
    if isinstance(obj, dict):
        if obj.get("class_name") == "DepthwiseConv2D" and "config" in obj and "name" in obj["config"]:
            depthwise_layers.add(obj["config"]["name"])
        for k, v in obj.items():
            collect_depthwise(v)
    elif isinstance(obj, list):
        for x in obj:
            collect_depthwise(x)

collect_depthwise(orig_data)
print(f"Found {len(depthwise_layers)} depthwise layers.")

# Let's write a fresh model.json starting from the original root model.json
# 1. Unwrap
if "modelTopology" in orig_data and "model_config" in orig_data["modelTopology"]:
    mc = orig_data["modelTopology"]["model_config"]
    if "class_name" in mc:
        orig_data["modelTopology"]["class_name"] = mc["class_name"]
        orig_data["modelTopology"]["config"] = mc["config"]
        orig_data["modelTopology"]["keras_version"] = "2.15.0"
        del orig_data["modelTopology"]["model_config"]

# 2. Fix Inbound Nodes FIRST (so we extract tensors from dicts)
def is_keras_tensor(obj):
    return isinstance(obj, dict) and obj.get("class_name") == "__keras_tensor__" and "config" in obj and "keras_history" in obj["config"]

def convert_tensor(obj):
    history = obj["config"]["keras_history"]
    return [history[0], history[1], history[2], {}]

def fix_inbound_nodes(nodes):
    new_nodes = []
    for node in nodes:
        if isinstance(node, dict):
            args = node.get("args", [])
            flat_args = []
            for arg in args:
                if isinstance(arg, list):
                    nested = [convert_tensor(x) for x in arg if is_keras_tensor(x)]
                    if nested:
                        flat_args.extend(nested)
                elif is_keras_tensor(arg):
                    flat_args.append(convert_tensor(arg))
            if flat_args:
                new_nodes.append(flat_args)
        else:
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

fix_all(orig_data)

# 3. Clean standard kwargs
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

orig_data = general_clean(orig_data)
orig_data["format"] = "layers-model"

# 4. FIX MANIFEST WEIGHT NAMES
# Strip `sequential/` from everything
# If the layer is in depthwise_layers, and the parameter is `kernel`, rename to `depthwise_kernel`
renamed_depthwise = 0
stripped_sequential = 0
for group in orig_data.get("weightsManifest", []):
    for w in group.get("weights", []):
        name = w["name"]
        if name.startswith("sequential/"):
            name = name[len("sequential/"):]
            stripped_sequential += 1
            
        parts = name.rsplit("/", 1)
        if len(parts) == 2:
            layer_name, param_name = parts
            if layer_name in depthwise_layers and param_name == "kernel":
                name = layer_name + "/depthwise_kernel"
                renamed_depthwise += 1
                
        w["name"] = name

print(f"Stripped sequential from {stripped_sequential} weights.")
print(f"Renamed {renamed_depthwise} kernel to depthwise_kernel.")

with open("public/AI/model.json", "w", encoding="utf-8") as f:
    json.dump(orig_data, f, indent=2)

print("Saved to public/AI/model.json")
