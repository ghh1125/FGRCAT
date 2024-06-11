import json
import networkx as nx
import matplotlib.pyplot as plt

def generate_relation_graph(file_path):
    G = nx.DiGraph()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            stem = data['question']['stem']
            question_concept = data['question']['question_concept']
            answer = data['answerKey']
            answer_text = data['question']['choices'][ord(answer) - ord('A')]['text']

            G.add_node(stem)
            G.add_node(question_concept)
            G.add_edge(stem, question_concept, label=answer_text)

    return G

def save_graph_to_file(G, file_path):
    data = nx.node_link_data(G)
    with open(file_path, 'w') as f:
        json.dump(data, f)

dev_graph = generate_relation_graph("./QA/dev_rand_split.jsonl")
train_graph = generate_relation_graph("./QA/train_rand_split.jsonl")

combined_graph = nx.compose(dev_graph, train_graph)
save_graph_to_file(combined_graph, 'QArelated.json')
