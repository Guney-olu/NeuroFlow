import networkx as nx
import matplotlib.pyplot as plt

def visualize_neural_network(nn):
    G = nx.DiGraph()
    input_nodes = list(range(nn.input_size))
    G.add_nodes_from(input_nodes, layer='input')
    hidden_nodes = list(range(nn.input_size, nn.input_size + nn.hidden_size))
    G.add_nodes_from(hidden_nodes, layer='hidden')
    G.add_edges_from([(i, j) for i in input_nodes for j in hidden_nodes])
    output_nodes = list(range(nn.input_size + nn.hidden_size, nn.input_size + nn.hidden_size + nn.output_size))
    G.add_nodes_from(output_nodes, layer='output')
    G.add_edges_from([(i, j) for i in hidden_nodes for j in output_nodes])

    pos = {
        **{node: (0, -index) for index, node in enumerate(input_nodes)},
        **{node: (1, -index) for index, node in enumerate(hidden_nodes)},
        **{node: (2, -index) for index, node in enumerate(output_nodes)}
    }

    layers = {'input': input_nodes, 'hidden': hidden_nodes, 'output': output_nodes}
    colors = {'input': 'blue', 'hidden': 'green', 'output': 'red'}
    node_colors = [colors[G.nodes[node]['layer']] for node in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
    edge_weights_input_hidden = {(i, j): round(weight, 2) for i, row in enumerate(nn.weights_input_hidden) for j, weight in enumerate(row)}
    edge_weights_hidden_output = {(i, j): round(weight, 2) for i, row in enumerate(nn.weights_hidden_output) for j, weight in enumerate(row)}
    edge_weights = {**edge_weights_input_hidden, **edge_weights_hidden_output}
    edge_colors = [plt.cm.Blues(weight) for weight in edge_weights.values()]  # Use plt.cm.Blues to get the colormap values
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2)
    
    edge_labels = {(i, j): f'{weight}' for (i, j), weight in edge_weights.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=8, verticalalignment='center')
    
    labels = {node: G.nodes[node]["layer"] for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color='white')

    plt.axis('off')
    plt.show()