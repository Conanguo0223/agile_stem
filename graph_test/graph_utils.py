def build_graph_from_info(graph_info):
    """
    Reconstruct graph from recorded graph_info
    """
    from torch_geometric.data import HeteroData
    import torch
    
    data = HeteroData()
    
    # Group nodes by type
    node_types = {}
    for node_name, node_data in graph_info['nodes'].items():
        node_type = node_data['type']
        if node_type not in node_types:
            node_types[node_type] = []
        node_types[node_type].append((node_name, node_data))
    
    # Create node features and mappings
    node_mapping = {}
    for node_type, nodes in node_types.items():
        features = []
        for i, (node_name, node_data) in enumerate(nodes):
            # Create feature vector from recorded data
            feat_vector = [
                node_data.get('processing_time', 0),
                node_data.get('position', [0, 0])[0] / 1000,
                node_data.get('position', [0, 0])[1] / 1000,
                node_data.get('carrier_capacity', 1),
            ]
            features.append(feat_vector)
            node_mapping[node_name] = (node_type, i)
        
        data[node_type].x = torch.tensor(features, dtype=torch.float)
    
    # Group and add edges
    edge_types = {}
    for edge_data in graph_info['edges']:
        source_name = edge_data['source']
        target_name = edge_data['target']
        
        source_type = node_mapping[source_name][0]
        target_type = node_mapping[target_name][0]
        
        edge_type = (source_type, 'connects_to', target_type)
        
        if edge_type not in edge_types:
            edge_types[edge_type] = []
        
        source_idx = node_mapping[source_name][1]
        target_idx = node_mapping[target_name][1]
        edge_types[edge_type].append([source_idx, target_idx])
    
    # Add edges to HeteroData
    for edge_type, edge_list in edge_types.items():
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            data[edge_type].edge_index = edge_index
    
    return data

# Usage
def get_line_graph(line, format='heterodata'):
    """
    Get graph representation from a line
    """
    if not hasattr(line, 'graph_info'):
        # If graph_info not recorded, extract from built components
        line.reset()  # This calls build() which should populate graph_info
    
    return build_graph_from_info(line.graph_info, output_format=format)