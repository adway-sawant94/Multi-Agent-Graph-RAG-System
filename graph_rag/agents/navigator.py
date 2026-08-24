import networkx as nx
from typing import List, Dict, Set, Any, Tuple
from graph_rag.agents.base import BaseAgent
from graph_rag.graph_store import GraphStore

class NavigatorAgent(BaseAgent):
    def __init__(self, graph_store: GraphStore):
        super().__init__("The Navigator", "Graph-Aware Traversal Agent")
        self.graph_store = graph_store

    def run(self, entry_entities: Set[str], depth: int = 2) -> Dict[str, Any]:
        """Traverses the bipartite/relational graph from the starting entity nodes.
        
        Collects triplets, chunks, and records the traversal paths.
        """
        self.log(f"Starting graph traversal from entry nodes: {list(entry_entities)} with depth={depth}")
        
        visited_nodes: Set[str] = set()
        retrieved_triplets: List[Dict[str, str]] = []
        retrieved_chunks: Set[str] = set()
        traversal_paths: List[List[str]] = []
        
        # We start search from each entry entity
        queue: List[Tuple[str, List[str], int]] = [(node, [node], 0) for node in entry_entities if self.graph_store.bipartite_graph.has_node(node)]
        
        for item in queue:
            node, path, curr_depth = item
            visited_nodes.add(node)
            
        idx = 0
        while idx < len(queue):
            curr_node, curr_path, curr_depth = queue[idx]
            idx += 1
            
            self.log(f"Navigating node '{curr_node}' (Depth {curr_depth})")
            
            # 1. Fetch text chunks associated with this node to ground the answer in raw text
            for neighbor in self.graph_store.bipartite_graph.neighbors(curr_node):
                n_data = self.graph_store.bipartite_graph.nodes[neighbor]
                if n_data.get("type") == "chunk":
                    retrieved_chunks.add(neighbor)
            
            # 2. Check depth constraint before exploring further
            if curr_depth >= depth:
                continue
                
            # 3. Traverse neighbor entities using relation-free flattening if node is too dense
            # Section 4.1 optimization
            neighbors = self.graph_store.relation_free_flatten(curr_node, max_degree=15)
            
            for neighbor, relationship in neighbors:
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    new_path = curr_path + [neighbor]
                    traversal_paths.append(new_path)
                    queue.append((neighbor, new_path, curr_depth + 1))
                    
                    # Capture the triplet representation
                    # Check edge data to see who is subject vs object if it's a triplet
                    edge_data = self.graph_store.bipartite_graph.get_edge_data(curr_node, neighbor)
                    if edge_data and edge_data.get("relationship") == "triplet":
                        predicate = edge_data.get("predicate", relationship)
                        retrieved_triplets.append({
                            "subject": curr_node,
                            "predicate": predicate,
                            "object": neighbor,
                            "source_chunk": edge_data.get("source_chunk", "")
                        })
                        
        self.log(f"Traversal finished. Visited {len(visited_nodes)} nodes.")
        self.log(f"Collected {len(retrieved_triplets)} RDF triplets.")
        self.log(f"Retrieved {len(retrieved_chunks)} supporting text chunks.")
        
        # Load raw chunk texts
        chunks_context = []
        for chunk_id in retrieved_chunks:
            chunk_data = self.graph_store.bipartite_graph.nodes[chunk_id]
            chunks_context.append({
                "id": chunk_id,
                "text": chunk_data.get("text", ""),
                "source": chunk_data.get("source_file", "")
            })
            
        return {
            "visited_entities": list(visited_nodes),
            "triplets": retrieved_triplets,
            "chunks": chunks_context,
            "paths": traversal_paths
        }
