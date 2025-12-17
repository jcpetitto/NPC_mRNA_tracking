import sys
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import csv

class CallGraphTracer:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.call_stack = []
        self.call_counts = defaultdict(int)
        self.node_attributes = defaultdict(dict)
        
    def tracer(self, frame, event, arg):
        if event == 'call':
            # Get function name and module
            func_name = frame.f_code.co_name
            module_name = frame.f_globals.get('__name__', 'unknown')
            file_name = frame.f_code.co_filename
            
            # Create unique node ID with module prefix
            node_id = f"{module_name}.{func_name}"
            
            # Add node with attributes
            if node_id not in self.graph:
                self.graph.add_node(node_id)
                self.node_attributes[node_id] = {
                    'function': func_name,
                    'module': module_name,
                    'file': file_name,
                    'call_count': 0
                }
            
            self.node_attributes[node_id]['call_count'] += 1
            
            # Add edge from caller to callee
            if self.call_stack:
                caller = self.call_stack[-1]
                self.graph.add_edge(caller, node_id)
                self.call_counts[(caller, node_id)] += 1
            
            self.call_stack.append(node_id)
            
        elif event == 'return':
            if self.call_stack:
                self.call_stack.pop()
                
        return self.tracer
    
    def start(self):
        sys.settrace(self.tracer)
        
    def stop(self):
        sys.settrace(None)
        
    def export_graphml(self, filename='callgraph.graphml'):
        """Export to GraphML format (recommended for Cytoscape)"""
        # Create a copy of the graph with attributes
        export_graph = self.graph.copy()
        
        # Add node attributes
        for node, attrs in self.node_attributes.items():
            for key, value in attrs.items():
                export_graph.nodes[node][key] = value
        
        # Add edge attributes (call counts)
        for (u, v), count in self.call_counts.items():
            if export_graph.has_edge(u, v):
                export_graph[u][v]['call_count'] = count
                export_graph[u][v]['weight'] = count
        
        # Write GraphML file
        nx.write_graphml(export_graph, filename)
        print(f"Exported call graph to {filename}")
    
    def export_gexf(self, filename='callgraph.gexf'):
        """Export to GEXF format (Gephi/Cytoscape compatible)"""
        export_graph = self.graph.copy()
        
        # Add attributes
        for node, attrs in self.node_attributes.items():
            for key, value in attrs.items():
                export_graph.nodes[node][key] = value
        
        for (u, v), count in self.call_counts.items():
            if export_graph.has_edge(u, v):
                export_graph[u][v]['weight'] = count
        
        nx.write_gexf(export_graph, filename)
        print(f"Exported call graph to {filename}")
    
    def export_cytoscape_json(self, filename='callgraph.json'):
        """Export to Cytoscape.js JSON format"""
        cytoscape_data = {
            'elements': {
                'nodes': [],
                'edges': []
            }
        }
        
        # Add nodes
        for node in self.graph.nodes():
            node_data = {
                'data': {
                    'id': node,
                    'label': node.split('.')[-1],  # Just function name for label
                    **self.node_attributes.get(node, {})
                }
            }
            cytoscape_data['elements']['nodes'].append(node_data)
        
        # Add edges
        for i, (source, target) in enumerate(self.graph.edges()):
            edge_data = {
                'data': {
                    'id': f'edge_{i}',
                    'source': source,
                    'target': target,
                    'weight': self.call_counts.get((source, target), 1),
                    'call_count': self.call_counts.get((source, target), 1)
                }
            }
            cytoscape_data['elements']['edges'].append(edge_data)
        
        with open(filename, 'w') as f:
            json.dump(cytoscape_data, f, indent=2)
        print(f"Exported call graph to {filename}")
    
    def export_csv(self, nodes_file='nodes.csv', edges_file='edges.csv'):
        """Export as CSV files (nodes and edges separately)"""
        # Export nodes
        with open(nodes_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'label', 'module', 'file', 'call_count'])
            
            for node in self.graph.nodes():
                attrs = self.node_attributes.get(node, {})
                writer.writerow([
                    node,
                    attrs.get('function', node),
                    attrs.get('module', ''),
                    attrs.get('file', ''),
                    attrs.get('call_count', 0)
                ])
        
        # Export edges
        with open(edges_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'target', 'weight', 'call_count'])
            
            for source, target in self.graph.edges():
                count = self.call_counts.get((source, target), 1)
                writer.writerow([source, target, count, count])
        
        print(f"Exported nodes to {nodes_file} and edges to {edges_file}")
    
    def export_gml(self, filename='callgraph.gml'):
        """Export to GML format"""
        export_graph = self.graph.copy()
        
        # Add attributes
        for node, attrs in self.node_attributes.items():
            for key, value in attrs.items():
                export_graph.nodes[node][key] = str(value)  # GML requires string values
        
        for (u, v), count in self.call_counts.items():
            if export_graph.has_edge(u, v):
                export_graph[u][v]['weight'] = count
        
        nx.write_gml(export_graph, filename)
        print(f"Exported call graph to {filename}")
    
    def export_all_formats(self, prefix='callgraph'):
        """Export to all supported formats"""
        self.export_graphml(f"{prefix}.graphml")
        self.export_cytoscape_json(f"{prefix}.json")
        self.export_csv(f"{prefix}_nodes.csv", f"{prefix}_edges.csv")
        self.export_gexf(f"{prefix}.gexf")
        self.export_gml(f"{prefix}.gml")
    
    def get_graph_stats(self):
        """Print statistics about the call graph"""
        print("\nCall Graph Statistics:")
        print(f"Total functions: {self.graph.number_of_nodes()}")
        print(f"Total calls: {self.graph.number_of_edges()}")
        print(f"Total call instances: {sum(self.call_counts.values())}")
        
        # Most called functions
        print("\nMost called functions:")
        sorted_nodes = sorted(self.node_attributes.items(), 
                            key=lambda x: x[1].get('call_count', 0), 
                            reverse=True)[:10]
        for node, attrs in sorted_nodes:
            print(f"  {node}: {attrs.get('call_count', 0)} calls")
        
        # Most frequent call paths
        print("\nMost frequent call paths:")
        sorted_edges = sorted(self.call_counts.items(), 
                            key=lambda x: x[1], 
                            reverse=True)[:10]
        for (source, target), count in sorted_edges:
            print(f"  {source} -> {target}: {count} calls")
    
    def draw(self, filename='callgraph.png'):
        """Original drawing method"""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', 
                             node_size=2000, alpha=0.9)
        
        # Draw edges with thickness based on call count
        edges = self.graph.edges()
        weights = [self.call_counts.get((u, v), 1) for u, v in edges]
        nx.draw_networkx_edges(self.graph, pos, width=weights, alpha=0.6, 
                             edge_color='gray', arrows=True)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=10)
        
        plt.title("Function Call Graph")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

# Example usage
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def process_math():
    return factorial(5) + fibonacci(5)

def main():
    result = process_math()
    print(f"Result: {result}")

# # Create and use tracer
# tracer = CallGraphTracer()
# tracer.start()
# main()
# tracer.stop()

# # Export to various formats
# tracer.export_all_formats('my_callgraph')

# # Or export to specific format for Cytoscape
# tracer.export_graphml('for_cytoscape.graphml')  # Recommended
# # tracer.export_cytoscape_json('for_cytoscape.json')  # Alternative

# # Print statistics
# tracer.get_graph_stats()

# # Still can use the original visualization
# tracer.draw()