import json
from typing import List, Dict, Set, Any, Tuple
from graph_rag.agents.base import BaseAgent
from graph_rag import llm, config
from graph_rag.vector_store import VectorStore
from graph_rag.graph_store import GraphStore

class MapperAgent(BaseAgent):
    def __init__(self, vector_store: VectorStore, graph_store: GraphStore):
        super().__init__("The Mapper", "Initial Query Entry Router")
        self.vector_store = vector_store
        self.graph_store = graph_store

    def run(self, query: str) -> Tuple[List[str], Set[str]]:
        """Maps user query to skeleton domains and returns a list of candidate entry entity nodes."""
        self.log(f"Received query: '{query}'")
        self.log("Retrieving matching document chunks from vector store...")
        
        # 1. Similarity Search over chunks
        query_emb = llm.get_embedding(query)
        similar_chunks = self.vector_store.similarity_search(query_emb, k=3)
        
        chunk_ids = [chunk[0] for chunk in similar_chunks]
        self.log(f"Retrieved top-3 chunks: {chunk_ids}")
        
        # 2. Extract entities connected to these chunks
        candidate_entities: Set[str] = set()
        for chunk_id in chunk_ids:
            if self.graph_store.bipartite_graph.has_node(chunk_id):
                # Look at neighbors of the chunk node (which are entities mentioned in it)
                for neighbor in self.graph_store.bipartite_graph.neighbors(chunk_id):
                    node_data = self.graph_store.bipartite_graph.nodes[neighbor]
                    if node_data.get("type") == "entity":
                        candidate_entities.add(neighbor)
                        
        self.log(f"Identified {len(candidate_entities)} entities linked to vector results: {list(candidate_entities)}")

        # 3. Route domains via LLM classification (Skeleton Routing)
        domains = self.graph_store.get_skeleton_domains()
        if not domains:
            # Fallback if no domains exist
            return [], candidate_entities
            
        system_instruction = (
            "You are the Routing Agent. Your job is to select which high-level domains are relevant to the user query. "
            "You must choose ONLY from the provided list of domains. "
            "Return the output in this JSON format: {'mapped_domains': ['Domain1', 'Domain2'], 'reason': 'Brief reasoning'}"
        )
        
        prompt = f"Available Domains:\n{domains}\n\nQuery:\n{query}\n\nSelect the relevant domains."
        
        mapped_domains = []
        try:
            raw_res = llm.generate_text(prompt, system_instruction=system_instruction)
            cleaned = raw_res.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            data = json.loads(cleaned)
            mapped_domains = data.get("mapped_domains", [])
            reason = data.get("reason", "No reason provided.")
            self.log(f"Mapped to skeleton domains: {mapped_domains} (Reason: {reason})")
        except Exception as e:
            self.log(f"LLM routing failed or mock mode triggered: {e}. Falling back to default domains.")
            # Basic fallback: match keywords in query
            for domain in domains:
                if domain.lower() in query.lower() or any(ent.lower() in domain.lower() for ent in candidate_entities):
                    mapped_domains.append(domain)
            if not mapped_domains:
                mapped_domains = [domains[0]]
                
        # 4. Expand starting entity nodes by including domain-associated entities
        for domain in mapped_domains:
            domain_ents = self.graph_store.get_entities_in_domain(domain)
            # Find overlaps to prioritize, but add domain entities as starting candidates
            self.log(f"Domain '{domain}' has entities: {domain_ents}")
            for ent in domain_ents:
                # Add to candidate entities if it is mentioned in any indexed chunks
                if self.graph_store.bipartite_graph.has_node(ent):
                    candidate_entities.add(ent)
                    
        self.log(f"Final routing completed. Mapped Domains: {mapped_domains}, Target Entry Entities: {list(candidate_entities)}")
        return mapped_domains, candidate_entities
