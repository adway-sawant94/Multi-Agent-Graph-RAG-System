import logging
from typing import Dict, Any, List, Tuple
from graph_rag.vector_store import VectorStore
from graph_rag.graph_store import GraphStore
from graph_rag.agents.mapper import MapperAgent
from graph_rag.agents.navigator import NavigatorAgent
from graph_rag.agents.synthesizer import SynthesizerAgent

logger = logging.getLogger("graph_rag.orchestrator")

class MultiAgentOrchestrator:
    def __init__(self, vector_store: VectorStore, graph_store: GraphStore):
        self.vector_store = vector_store
        self.graph_store = graph_store
        
        # Instantiate Agents
        self.mapper = MapperAgent(vector_store, graph_store)
        self.navigator = NavigatorAgent(graph_store)
        self.synthesizer = SynthesizerAgent()

    def query(self, query_text: str, traversal_depth: int = 2) -> Dict[str, Any]:
        """Runs the entire Multi-Agent Graph-RAG pipeline.
        
        Ties together the Mapper, Navigator, and Synthesizer agents.
        """
        logger.info(f"Initiating Multi-Agent Graph-RAG Query: '{query_text}'")
        
        # Clear previous logs for clean execution tracking
        self.mapper.clear_logs()
        self.navigator.clear_logs()
        self.synthesizer.clear_logs()
        
        # Step 1: Mapping Phase (Skeleton Routing & Entry point identification)
        mapped_domains, entry_entities = self.mapper.run(query_text)
        
        if not entry_entities:
            # If no starting entities found, default to direct chunk retrieval
            logger.warning("No entry entities mapped from skeleton or vector store.")
            
        # Step 2: Navigation Phase (Multi-Hop Graph Traversal)
        traversal_data = self.navigator.run(entry_entities, depth=traversal_depth)
        
        # Step 3: Synthesis Phase (Map-Reduce answer compilation)
        answer, audit_trace = self.synthesizer.run(query_text, traversal_data)
        
        # Combine all agent logs chronologically
        combined_logs = []
        combined_logs.append("=== PHASE 1: QUERY ROUTING (The Mapper) ===")
        combined_logs.extend(self.mapper.get_logs())
        combined_logs.append("\n=== PHASE 2: GRAPH TRAVERSAL (The Navigator) ===")
        combined_logs.extend(self.navigator.get_logs())
        combined_logs.append("\n=== PHASE 3: RESPONSE GENERATION (The Synthesizer) ===")
        combined_logs.extend(self.synthesizer.get_logs())
        
        logger.info("Multi-Agent Graph-RAG Pipeline execution complete.")
        
        return {
            "answer": answer,
            "audit_trace": audit_trace,
            "agent_logs": combined_logs,
            "mapped_domains": mapped_domains,
            "entry_entities": list(entry_entities),
            "visited_entities": traversal_data["visited_entities"],
            "retrieved_triplets": traversal_data["triplets"],
            "retrieved_chunks": traversal_data["chunks"],
            "paths": traversal_data["paths"]
        }
