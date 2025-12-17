import sys
import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import csv

class CallGraphTracer:
    def __init__(self, scope_directory=None, include_subdirs=True):
        self.graph = nx.DiGraph()
        self.call_stack = []
        self.call_counts = defaultdict(int)
        self.node_attributes = defaultdict(dict)
        self.include_subdirs = include_subdirs
        
        # Determine scope directory
        if scope_directory:
            self.scope_directory = os.path.abspath(scope_directory)
        else:
            # Get the directory of the main script
            import __main__
            if hasattr(__main__, '__file__'):
                self.scope_directory = os.path.dirname(os.path.abspath(__main__.__file__))
            else:
                # If running in interactive mode, use current directory
                self.scope_directory = os.path.abspath(os.getcwd())
        
        print(f"Tracing calls within scope: {self.scope_directory}")
        
    def is_in_scope(self, file_path):
        """Check if a file is within the scope directory"""
        if not file_path:
            return False
            
        # Normalize the path
        abs_path = os.path.abspath(file_path)
        
        # Skip built-in modules (no real file path)
        if not os.path.exists(abs_path):
            return False
        
        # Check if file is in scope directory
        if self.include_subdirs:
            return abs_path.startswith(self.scope_directory)
        else:
            return os.path.dirname(abs_path) == self.scope_directory
    
    def tracer(self, frame, event, arg):
        if event == 'call':
            # Get function details
            func_name = frame.f_code.co_name
            module_name = frame.f_globals.get('__name__', 'unknown')
            file_path = frame.f_code.co_filename
            
            # Skip if outside scope
            if not self.is_in_scope(file_path):
                return self.tracer
            
            # Skip internal/private functions if desired
            if func_name.startswith('_'):
                return self.tracer
            
            # Create node ID
            # Use relative path for cleaner visualization
            try:
                rel_path = os.path.relpath(file_path, self.scope_directory)
            except ValueError:
                rel_path = file_path
            
            node_id = f"{module_name}.{func_name}"
            
            # Add node with attributes
            if node_id not in self.graph:
                self.graph.add_node(node_id)
                self.node_attributes[node_id] = {
                    'function': func_name,
                    'module': module_name,
                    'file': file_path,
                    'relative_file': rel_path,
                    'call_count': 0,
                    'line_number': frame.f_code.co_firstlineno
                }
            
            self.node_attributes[node_id]['call_count'] += 1
            
            # Add edge only if caller is also in scope
            if self.call_stack:
                # Find the last in-scope caller
                for caller in reversed(self.call_stack):
                    if caller in self.graph:  # Caller is in scope
                        self.graph.add_edge(caller, node_id)
                        self.call_counts[(caller, node_id)] += 1
                        break
            
            self.call_stack.append(node_id)
            
        elif event == 'return':
            if self.call_stack:
                self.call_stack.pop()
                
        return self.tracer
    
    def start(self):
        sys.settrace(self.tracer)
        
    def stop(self):
        sys.settrace(None)
    
    def filter_external_calls(self):
        """Remove any remaining external calls that might have slipped through"""
        nodes_to_remove = []
        for node in self.graph.nodes():
            attrs = self.node_attributes.get(node, {})
            if 'file' in attrs and not self.is_in_scope(attrs['file']):
                nodes_to_remove.append(node)
        
        for node in nodes_to_remove:
            self.graph.remove_node(node)
            if node in self.node_attributes:
                del self.node_attributes[node]
    
    def export_graphml(self, filename='callgraph.graphml'):
        """Export to GraphML format (recommended for Cytoscape)"""
        # Apply final filter
        self.filter_external_calls()
        
        export_graph = self.graph.copy()
        
        # Add node attributes
        for node, attrs in self.node_attributes.items():
            for key, value in attrs.items():
                export_graph.nodes[node][key] = value
        
        # Add edge attributes
        for (u, v), count in self.call_counts.items():
            if export_graph.has_edge(u, v):
                export_graph[u][v]['call_count'] = count
                export_graph[u][v]['weight'] = count
        
        nx.write_graphml(export_graph, filename)
        print(f"Exported call graph to {filename}")
        print(f"  Nodes: {export_graph.number_of_nodes()}")
        print(f"  Edges: {export_graph.number_of_edges()}")
    
    def export_cytoscape_json(self, filename='callgraph.json'):
        """Export to Cytoscape.js JSON format"""
        # Apply final filter
        self.filter_external_calls()
        
        cytoscape_data = {
            'elements': {
                'nodes': [],
                'edges': []
            },
            'data': {
                'scope_directory': self.scope_directory
            }
        }
        
        # Add nodes
        for node in self.graph.nodes():
            attrs = self.node_attributes.get(node, {})
            node_data = {
                'data': {
                    'id': node,
                    'label': attrs.get('function', node.split('.')[-1]),
                    'module': attrs.get('module', ''),
                    'file': attrs.get('relative_file', ''),
                    'call_count': attrs.get('call_count', 0),
                    'line_number': attrs.get('line_number', 0)
                }
            }
            cytoscape_data['elements']['nodes'].append(node_data)
        
        # Add edges
        edge_id = 0
        for source, target in self.graph.edges():
            if source in self.graph and target in self.graph:  # Both nodes are in scope
                edge_data = {
                    'data': {
                        'id': f'edge_{edge_id}',
                        'source': source,
                        'target': target,
                        'weight': self.call_counts.get((source, target), 1)
                    }
                }
                cytoscape_data['elements']['edges'].append(edge_data)
                edge_id += 1
        
        with open(filename, 'w') as f:
            json.dump(cytoscape_data, f, indent=2)
        print(f"Exported call graph to {filename}")
    
    def export_csv(self, nodes_file='nodes.csv', edges_file='edges.csv'):
        """Export as CSV files"""
        # Apply final filter
        self.filter_external_calls()
        
        # Export nodes
        with open(nodes_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'function', 'module', 'relative_file', 'call_count', 'line_number'])
            
            for node in self.graph.nodes():
                attrs = self.node_attributes.get(node, {})
                writer.writerow([
                    node,
                    attrs.get('function', ''),
                    attrs.get('module', ''),
                    attrs.get('relative_file', ''),
                    attrs.get('call_count', 0),
                    attrs.get('line_number', 0)
                ])
        
        # Export edges
        with open(edges_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'target', 'weight'])
            
            for source, target in self.graph.edges():
                count = self.call_counts.get((source, target), 1)
                writer.writerow([source, target, count])
        
        print(f"Exported nodes to {nodes_file} and edges to {edges_file}")
    
    def get_graph_stats(self):
        """Print statistics about the call graph"""
        # Apply final filter
        self.filter_external_calls()
        
        print("\nCall Graph Statistics:")
        print(f"Scope directory: {self.scope_directory}")
        print(f"Include subdirectories: {self.include_subdirs}")
        print(f"Total functions in scope: {self.graph.number_of_nodes()}")
        print(f"Total calls tracked: {self.graph.number_of_edges()}")
        print(f"Total call instances: {sum(self.call_counts.values())}")
        
        # Files represented
        files = set()
        for attrs in self.node_attributes.values():
            if 'relative_file' in attrs:
                files.add(attrs['relative_file'])
        print(f"Files in scope: {len(files)}")
        for f in sorted(files):
            print(f"  - {f}")
        
        # Most called functions
        print("\nMost called functions:")
        sorted_nodes = sorted(self.node_attributes.items(), 
                            key=lambda x: x[1].get('call_count', 0), 
                            reverse=True)[:10]
        for node, attrs in sorted_nodes:
            print(f"  {node}: {attrs.get('call_count', 0)} calls")
    
    def draw(self, filename='callgraph.png', layout='spring'):
        """Enhanced drawing method with layout options"""
        # Apply final filter
        self.filter_external_calls()
        
        plt.figure(figsize=(14, 10))
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, k=3, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        elif layout == 'hierarchical':
            # Try to create a hierarchical layout
            try:
                pos = nx.nx_agraph.graphviz_layout(self.graph, prog='dot')
            except:
                print("Hierarchical layout requires pygraphviz. Using spring layout instead.")
                pos = nx.spring_layout(self.graph, k=3, iterations=50)
        else:
            pos = nx.spring_layout(self.graph, k=3, iterations=50)
        
        # Node sizes based on call count
        node_sizes = [self.node_attributes.get(node, {}).get('call_count', 1) * 300 
                     for node in self.graph.nodes()]
        
        # Color nodes by module/file
        module_colors = {}
        color_idx = 0
        node_colors = []
        for node in self.graph.nodes():
            module = self.node_attributes.get(node, {}).get('module', 'unknown')
            if module not in module_colors:
                module_colors[module] = plt.cm.tab20(color_idx % 20)
                color_idx += 1
            node_colors.append(module_colors[module])
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, 
                             node_color=node_colors,
                             node_size=node_sizes, 
                             alpha=0.8)
        
        # Draw edges with thickness based on call count
        edges = self.graph.edges()
        weights = [min(self.call_counts.get((u, v), 1), 10) for u, v in edges]
        nx.draw_networkx_edges(self.graph, pos, 
                             width=weights, 
                             alpha=0.5,
                             edge_color='gray', 
                             arrows=True,
                             arrowsize=15)
        
        # Draw labels
        labels = {node: node.split('.')[-1] for node in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        plt.title(f"Function Call Graph\nScope: {os.path.basename(self.scope_directory)}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()


# Example usage with a specific directory scope
if __name__ == "__main__":
    # Example functions
    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n-1)
    
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    def main():
        # This would normally call various functions
        import math  # This won't be tracked (external module)
        
        result1 = factorial(5)
        result2 = fibonacci(5)
        result3 = math.sqrt(16)  # External call - won't be tracked
        
        print(f"Results: {result1}, {result2}, {result3}")
    
    # Create tracer with current directory as scope
    tracer = CallGraphTracer(
        scope_directory=None,  # Uses main script's directory
        include_subdirs=True   # Include subdirectories in scope
    )
    
    