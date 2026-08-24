import json
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set, Optional

class GraphStore:
    def __init__(self):
        # Top-layer Knowledge Graph Skeleton (Directed)
        self.skeleton_graph = nx.DiGraph()
        
        # Lower-layer Text-Keyword Bipartite / Relational Graph (Undirected)
        self.bipartite_graph = nx.Graph()
        
        # Mapping of Skeleton Domains to Entities (bridges the two layers)
        self.domain_to_entities: Dict[str, Set[str]] = {}
        self.entity_to_domains: Dict[str, Set[str]] = {}

    # --- Skeleton Graph (Top Layer) ---
    
    def add_skeleton_node(self, node_id: str, category: str = "Domain", description: str = "") -> None:
        """Adds a high-level domain or topic node to the skeleton graph."""
        self.skeleton_graph.add_node(
            node_id, 
            category=category, 
            description=description,
            type="skeleton"
        )
        if node_id not in self.domain_to_entities:
            self.domain_to_entities[node_id] = set()

    def add_skeleton_edge(self, source: str, target: str, relationship: str = "related_to") -> None:
        """Adds a relationship between two macro domains."""
        self.skeleton_graph.add_edge(source, target, relationship=relationship)

    def get_skeleton_domains(self) -> List[str]:
        """Returns all macro domain nodes in the skeleton."""
        return list(self.skeleton_graph.nodes())

    # --- Bipartite/Relational Graph (Lower Layer) ---
    
    def add_chunk(self, chunk_id: str, text: str, source_file: str, doc_type: str = "text") -> None:
        """Adds a document text chunk node to the bipartite graph."""
        self.bipartite_graph.add_node(
            chunk_id,
            type="chunk",
            text=text,
            source_file=source_file,
            doc_type=doc_type
        )

    def add_entity(self, entity_id: str, entity_type: str = "Concept", description: str = "") -> None:
        """Adds an entity node (Subject/Object) to the bipartite graph."""
        # Clean entity ID to maintain standardization
        entity_id_clean = entity_id.strip()
        if self.bipartite_graph.has_node(entity_id_clean):
            # Update description if empty or extend
            existing_desc = self.bipartite_graph.nodes[entity_id_clean].get("description", "")
            if not existing_desc and description:
                self.bipartite_graph.nodes[entity_id_clean]["description"] = description
        else:
            self.bipartite_graph.add_node(
                entity_id_clean,
                type="entity",
                entity_type=entity_type,
                description=description
            )

    def add_entity_chunk_link(self, entity_id: str, chunk_id: str) -> None:
        """Creates a link indicating that an entity is mentioned in a text chunk."""
        entity_id_clean = entity_id.strip()
        if not self.bipartite_graph.has_node(entity_id_clean):
            self.add_entity(entity_id_clean)
        self.bipartite_graph.add_edge(entity_id_clean, chunk_id, relationship="mentioned_in")

    def add_triplet(self, subject: str, predicate: str, obj: str, source_chunk_id: Optional[str] = None) -> None:
        """Adds an RDF triplet (Subject -> Predicate -> Object) to the graph.
        
        This connects two entity nodes directly via an edge labeled with the predicate.
        """
        sub_clean = subject.strip()
        obj_clean = obj.strip()
        pred_clean = predicate.strip()
        
        # Ensure subject and object nodes exist
        if not self.bipartite_graph.has_node(sub_clean):
            self.add_entity(sub_clean)
        if not self.bipartite_graph.has_node(obj_clean):
            self.add_entity(obj_clean)
            
        # Add edge between entities
        # Since bipartite_graph is undirected but triplets are directed, we store the direction in the edge attributes
        self.bipartite_graph.add_edge(
            sub_clean, 
            obj_clean, 
            relationship="triplet",
            predicate=pred_clean,
            source_chunk=source_chunk_id or ""
        )
        
        # Link entities to the source chunk if provided
        if source_chunk_id:
            self.add_entity_chunk_link(sub_clean, source_chunk_id)
            self.add_entity_chunk_link(obj_clean, source_chunk_id)

    # --- Cross-Layer Bridging ---
    
    def link_entity_to_domain(self, entity_id: str, domain_id: str) -> None:
        """Links a low-level entity to a high-level skeleton domain."""
        entity_clean = entity_id.strip()
        if domain_id not in self.skeleton_graph:
            self.add_skeleton_node(domain_id)
            
        if domain_id not in self.domain_to_entities:
            self.domain_to_entities[domain_id] = set()
        self.domain_to_entities[domain_id].add(entity_clean)
        
        if entity_clean not in self.entity_to_domains:
            self.entity_to_domains[entity_clean] = set()
        self.entity_to_domains[entity_clean].add(domain_id)

    def get_entities_in_domain(self, domain_id: str) -> List[str]:
        """Returns all entities associated with a specific macro domain."""
        return list(self.domain_to_entities.get(domain_id, set()))

    # --- Graph Traversal and Search ---
    
    def get_rdf_triplets(self) -> List[Dict[str, str]]:
        """Returns all RDF triplets stored in the bipartite graph."""
        triplets = []
        for u, v, data in self.bipartite_graph.edges(data=True):
            if data.get("relationship") == "triplet":
                triplets.append({
                    "subject": u,
                    "predicate": data.get("predicate", "related_to"),
                    "object": v,
                    "source_chunk": data.get("source_chunk", "")
                })
        return triplets

    def relation_free_flatten(self, node: str, max_degree: int = 10) -> List[Tuple[str, str]]:
        """Optimizes traversal by returning a direct flattened list of neighbors
        for extremely dense clusters, bypassing step-by-step edge checking.
        
        This directly addresses Section 4.1 in the research paper.
        """
        if not self.bipartite_graph.has_node(node):
            return []
            
        degree = self.bipartite_graph.degree(node)
        neighbors = []
        
        if degree > max_degree:
            # Flattening: Directly extract entity neighbors without checking predicates
            for neighbor in self.bipartite_graph.neighbors(node):
                n_type = self.bipartite_graph.nodes[neighbor].get("type", "")
                if n_type == "entity":
                    neighbors.append((neighbor, "flattened_relationship"))
        else:
            # Traditional relationship traversal
            for neighbor in self.bipartite_graph.neighbors(node):
                edge_data = self.bipartite_graph.get_edge_data(node, neighbor)
                if edge_data.get("relationship") == "triplet":
                    predicate = edge_data.get("predicate", "connects")
                    neighbors.append((neighbor, predicate))
                    
        return neighbors

    # --- Serialization ---
    
    def save(self, filepath: Path) -> None:
        """Saves the skeleton graph and bipartite graph to a unified JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Serialize skeleton graph
        skeleton_nodes = list(self.skeleton_graph.nodes(data=True))
        skeleton_edges = list(self.skeleton_graph.edges(data=True))
        
        # Serialize bipartite graph
        bipartite_nodes = list(self.bipartite_graph.nodes(data=True))
        # Convert edge tuples (u, v, data) to serializable dicts
        bipartite_edges = []
        for u, v, data in self.bipartite_graph.edges(data=True):
            bipartite_edges.append({
                "source": u,
                "target": v,
                **data
            })
            
        # Convert sets to lists for JSON compatibility
        domain_mappings = {k: list(v) for k, v in self.domain_to_entities.items()}
        
        data = {
            "skeleton": {
                "nodes": skeleton_nodes,
                "edges": skeleton_edges
            },
            "bipartite": {
                "nodes": bipartite_nodes,
                "edges": bipartite_edges
            },
            "domain_mappings": domain_mappings
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: Path) -> None:
        """Loads a unified graph store from a JSON file."""
        filepath = Path(filepath)
        if not filepath.exists():
            return
            
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Reconstruct skeleton graph (Directed)
        self.skeleton_graph = nx.DiGraph()
        for node_id, attrs in data["skeleton"]["nodes"]:
            self.skeleton_graph.add_node(node_id, **attrs)
        for u, v, attrs in data["skeleton"]["edges"]:
            self.skeleton_graph.add_edge(u, v, **attrs)
            
        # Reconstruct bipartite graph (Undirected)
        self.bipartite_graph = nx.Graph()
        for node_id, attrs in data["bipartite"]["nodes"]:
            self.bipartite_graph.add_node(node_id, **attrs)
        for edge_data in data["bipartite"]["edges"]:
            u = edge_data.pop("source")
            v = edge_data.pop("target")
            self.bipartite_graph.add_edge(u, v, **edge_data)
            
        # Reconstruct domain mappings
        self.domain_to_entities = {k: set(v) for k, v in data["domain_mappings"].items()}
        self.entity_to_domains = {}
        for dom, ents in self.domain_to_entities.items():
            for ent in ents:
                if ent not in self.entity_to_domains:
                    self.entity_to_domains[ent] = set()
                self.entity_to_domains[ent].add(dom)
                
    def __len__(self) -> int:
        # Returns number of entity nodes in bipartite graph
        return len([n for n, d in self.bipartite_graph.nodes(data=True) if d.get("type") == "entity"])
