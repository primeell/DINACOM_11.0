import json

with open("public/AI/model.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def check_arrays(obj, path=""):
    if isinstance(obj, dict):
        for k in ["input_layers", "output_layers", "inbound_nodes"]:
            if k in obj:
                arr = obj[k]
                for i, item in enumerate(arr):
                    if isinstance(item, dict):
                        print(f"FOUND DICT IN {k} AT {path}[{i}]: {item}")
                    elif isinstance(item, list):
                        for j, inner in enumerate(item):
                            if isinstance(inner, dict):
                                print(f"FOUND DICT IN {k} AT {path}[{i}][{j}]: {inner}")
                            elif isinstance(inner, list):
                                for x, deeply in enumerate(inner):
                                    if isinstance(deeply, dict):
                                        print(f"FOUND DICT IN {k} AT {path}[{i}][{j}][{x}]: {deeply}")

        for k, v in obj.items():
            check_arrays(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, x in enumerate(obj):
            check_arrays(x, f"{path}[{i}]")

check_arrays(data, "root")
