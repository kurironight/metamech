import numpy as np


def make_main_node_edge_info(origin_nodes_positions, origin_edges_indices, origin_input_nodes, origin_input_vectors,
                             origin_output_nodes, origin_output_vectors, origin_frozen_nodes):

    new_input_nodes = np.arange(len(origin_input_nodes))
    new_output_nodes = np.arange(
        len(origin_output_nodes))+len(origin_input_nodes)
    new_frozen_nodes = np.arange(
        len(origin_frozen_nodes))+len(origin_input_nodes)+len(origin_output_nodes)

    new_node_pos = origin_nodes_positions[np.concatenate(
        [origin_input_nodes, origin_output_nodes, origin_frozen_nodes])]

    new_edges_indices = make_main_edges_indices(
        origin_edges_indices, origin_input_nodes, origin_output_nodes, origin_frozen_nodes)

    new_edges_thickness = np.ones(len(new_edges_indices))

    new_input_vectors = origin_input_vectors
    new_output_vectors = origin_output_vectors

    return new_node_pos, new_input_nodes, new_input_vectors, new_output_nodes, new_output_vectors, new_frozen_nodes, new_edges_indices, new_edges_thickness


def make_main_edges_indices(edges_indices, input_nodes, output_nodes, frozen_nodes):
    valid_nodes = np.concatenate([input_nodes, output_nodes, frozen_nodes])
    # edges_indicesのうち，初期エッジ情報のみを抽出する
    edges_bool = np.isin(edges_indices, valid_nodes)
    edges_bool = edges_bool[:, 0] & edges_bool[:, 1]
    edges_indices = edges_indices[edges_bool]

    new_edges_indices = np.zeros(shape=edges_indices.shape, dtype=int)
    for i, v in enumerate(valid_nodes):
        mask = edges_indices == v
        new_edges_indices[mask] = i

    return new_edges_indices
