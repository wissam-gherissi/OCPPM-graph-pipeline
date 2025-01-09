import networkx as nx
import pm4py
import torch
from networkx import relabel_nodes
from ocpa.algo.predictive_monitoring import factory as predictive_monitoring
from torch_geometric.data import Data

def ocel_to_csv(file_path, new_file_path, file_type):
    # Importing file and transforming into csv
    if file_type == "xml":
        ocel = pm4py.read_ocel2_xml(file_path)
    elif file_type == "jsonocel":
        ocel = pm4py.read_ocel(file_path)
    elif file_type == "sqlite":
        ocel = pm4py.read_ocel2_sqlite(file_path)
    df = ocel.get_extended_table()
    df.columns = [el.split(':')[-1] for el in df.columns]
    df.columns = [el.replace(' ', '_') for el in df.columns]
    df.to_csv(new_file_path)
    df.head()


def get_process_executions_nx(ocel, event_based_features, event_based_targets, execution_based_features,
                              execution_based_targets):
    feature_storage = predictive_monitoring.apply(ocel, event_based_features + event_based_targets,
                                                  execution_based_features + execution_based_targets)
    process_executions_nx = []
    for g in feature_storage.feature_graphs:
        nx_graph = nx.Graph()
        for edge in g.edges:
            nx_graph.add_edge(edge.source, edge.target)
        nx.set_node_attributes(nx_graph, {n.event_id: n.attributes for n in g.nodes})
        nx_graph.node_attr_dict_factory = event_based_features
        ts_pairs = [(idx.event_id, ocel.get_value(idx.event_id, "event_timestamp")) for idx in g.nodes]
        ts_pairs.sort(key=lambda x: x[1])
        sorted_idxs = [p[0] for p in ts_pairs]
        process_executions_nx.append((nx_graph, sorted_idxs, g.attributes))

    return process_executions_nx


def remaining_time(sub_pe):
    """
    Calculate remaining time from the last event in the subgraph and removes the target from the feature set

    Note: To remove a node attribute feature could also use:
        del sub_pe.nodes[i][('event_remaining_time', ())] in a for-loop
    """

    return min([sub_pe.nodes[e].pop(('event_remaining_time', ())) for e in sub_pe.nodes])


def num_events(pe_attributes):
    return pe_attributes[('num_events', ())]


def num_objects(pe_attributes):
    return pe_attributes[('exec_objects', ())]


def num_unique_acts(pe_attributes):
    return pe_attributes[('exec_uniq_activities', ())]


def get_subgraphs_labeled(PE_nx, k=8, subg_funcs=[remaining_time], g_funcs=[]):
    result = []
    for pe_id, (pe, sorted_idxs, pe_attributes) in enumerate(PE_nx):
        g_outs = []
        for g_func in g_funcs:
            g_outs.append(g_func(pe_attributes))
        for start in range(len(sorted_idxs) - k):
            subgraph = nx.subgraph(pe, sorted_idxs[start:start + k]).copy()
            node_mapping = {node: i for i, node in enumerate(subgraph.nodes)}
            subgraph = relabel_nodes(subgraph, node_mapping)
            subg_outs = []
            for subg_func in subg_funcs:
                out = subg_func(subgraph)
                subg_outs.append(out)
            subgraph.node_attr_dict_factory = pe.node_attr_dict_factory
            result.append([pe_id, subgraph] + subg_outs + g_outs)
    return result


def get_node_features(graph):
    node_features = []
    for node in graph.nodes:
        features = []
        for att_name in graph.node_attr_dict_factory:
            features.append(graph.nodes[node][att_name])
        node_features.append(features)

    # Convert to PyTorch tensor for input to GNN
    x = torch.tensor(node_features, dtype=torch.float)
    return x


def get_edge_index(graph):
    node_id_map = {gid: i for i, gid in enumerate(graph.nodes)}
    edge_index = torch.tensor([[node_id_map[e[0]] for e in graph.edges],
                               [node_id_map[e[1]] for e in graph.edges]])
    return edge_index


def generate_matrix_dataset(labeled_subgraphs):
    data_list = []
    for el in labeled_subgraphs:
        gid = el[0]
        subgraph = el[1]
        if len(subgraph.edges) > 0:
            y = el[2:]
            x = get_node_features(subgraph)
            edge_index = get_edge_index(subgraph)
            d = Data(id=gid, graph=subgraph, x=x, edge_index=edge_index, y=y)
            data_list.append(d)
    return data_list
