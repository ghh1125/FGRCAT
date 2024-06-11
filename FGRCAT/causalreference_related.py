import json
import networkx as nx
import matplotlib.pyplot as plt

def create_causal_graph(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    G = nx.DiGraph()

    for line in data:
        item = json.loads(line.strip())
        cause = item['cause']
        effect = item['effect']
        conceptual_explanation = item.get('conceptual_explanation', None)

        # Adding cause and effect as nodes
        G.add_node(cause)
        G.add_node(effect)

        # Adding edges from cause to conceptual_explanation and from conceptual_explanation to effect
        if conceptual_explanation:
            G.add_edge(cause, conceptual_explanation)
            G.add_edge(conceptual_explanation, effect)
        else:
            # If no conceptual explanation provided, connect cause directly to effect
            G.add_edge(cause, effect)

    return G

def save_graph_to_file(graph, file_name):
    data = nx.readwrite.json_graph.node_link_data(graph)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

dev_graph = create_causal_graph('/home/amax/ghh/e-CARE-main/dataset/COPA/processed_devEG.jsonl')
train_graph = create_causal_graph('/home/amax/ghh/e-CARE-main/dataset/COPA/processed_trainEG.jsonl')
combined_graph = nx.compose(dev_graph, train_graph)
save_graph_to_file(combined_graph, 'COPACIrelated.json')



