import json

with open("public/AI/model.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def find_bad_nodes(obj, path=""):
    if isinstance(obj, dict):
        if "inbound_nodes" in obj:
            nodes = obj["inbound_nodes"]
            if not isinstance(nodes, list):
                print(f"inbound_nodes is not list at {path}.name={obj.get('name')}: {nodes}")
            else:
                for icls, n in enumerate(nodes):
                    if isinstance(n, dict):
                        print(f"Found dict in inbound_nodes at {path}.name={obj.get('name')}[{icls}]: {n.keys()}")
                    elif isinstance(n, list):
                        for j, inner in enumerate(n):
                            if isinstance(inner, dict):
                                print(f"Found dict in inbound_nodes nested at {path}.name={obj.get('name')}[{icls}][{j}]: {inner.keys()}")
        
        for k, v in obj.items():
            find_bad_nodes(v, path + "." + k)
    elif isinstance(obj, list):
        for i, x in enumerate(obj):
            find_bad_nodes(x, path + f"[{i}]")

find_bad_nodes(data, "root")
