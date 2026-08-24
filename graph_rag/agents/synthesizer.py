from typing import List, Dict, Any, Tuple
from graph_rag.agents.base import BaseAgent
from graph_rag import llm

class SynthesizerAgent(BaseAgent):
    def __init__(self):
        super().__init__("The Synthesizer", "Information Aggregator and Answer Generator")

    def map_summarize(self, chunks: List[Dict[str, Any]], query: str) -> List[str]:
        """Summarizes each text chunk individually relative to the query (Map step)."""
        summaries = []
        for idx, chunk in enumerate(chunks):
            self.log(f"Summarizing context chunk {idx+1}/{len(chunks)} from source '{chunk['source']}'")
            
            system_instruction = (
                "You are an assistant summarizing corporate documents. "
                "Highlight only facts relevant to the query. Keep it extremely concise (1-2 sentences)."
            )
            
            prompt = f"Query: {query}\n\nDocument Chunk ({chunk['id']}):\n{chunk['text']}\n\nSummary:"
            
            summary = llm.generate_text(prompt, system_instruction=system_instruction)
            summaries.append(f"Source [{chunk['source']}]: {summary.strip()}")
            
        return summaries

    def run(self, query: str, traversal_data: Dict[str, Any]) -> Tuple[str, str]:
        """Compiles the final answer by combining the mapped summaries and RDF triplets (Reduce step)."""
        self.log("Beginning synthesis of retrieved graph data...")
        
        triplets = traversal_data.get("triplets", [])
        chunks = traversal_data.get("chunks", [])
        paths = traversal_data.get("paths", [])
        
        # 1. Map step: Summarize chunk texts
        mapped_summaries = []
        if chunks:
            mapped_summaries = self.map_summarize(chunks, query)
        else:
            self.log("No supporting text chunks retrieved.")
            
        # 2. Format RDF triplets for inclusion in LLM prompt
        triplet_strings = []
        for t in triplets:
            triplet_strings.append(f"({t['subject']}) --[{t['predicate']}]--> ({t['object']})")
            
        self.log(f"Formulated {len(triplet_strings)} explicit relationships for reasoning.")
        
        # 3. Create the final synthesis prompt (Reduce step)
        triplet_context = "\n".join(triplet_strings) if triplet_strings else "No explicit relations found."
        chunks_context = "\n".join(mapped_summaries) if mapped_summaries else "No raw text chunks found."
        
        system_instruction = (
            "You are an expert Synthesizer Agent. Your job is to answer the user query based ONLY on the provided "
            "factual context and semantic relationships. Do not hallucinate or assume facts not supported by the context.\n\n"
            "Format your answer beautifully using Markdown. You MUST explain the relationship chain "
            "retrieved from the graph to make your reasoning transparent and traceable."
        )
        
        prompt = (
            f"Query: {query}\n\n"
            f"=== RETRIEVED SEMANTIC RELATIONSHIPS (RDF Triplets) ===\n"
            f"{triplet_context}\n\n"
            f"=== RETRIEVED DOCUMENT SUMMARIES ===\n"
            f"{chunks_context}\n\n"
            f"Draft a comprehensive, professional answer that directly answers the query. "
            f"Make sure to clearly outline the connection path discovered in the graph."
        )
        
        self.log("Generating final grounded answer...")
        answer = llm.generate_text(prompt, system_instruction=system_instruction)
        
        # 4. Generate the audit trace description
        trace_steps = []
        trace_steps.append("### 🔍 Traceability & Auditing Path")
        trace_steps.append("The Navigator Agent traversed the following relationship paths to answer this query:")
        
        if paths:
            for p in paths[:5]:  # Limit to top 5 paths for readability
                path_str = " ➔ ".join(f"`{node}`" for node in p)
                trace_steps.append(f"- {path_str}")
        else:
            trace_steps.append("- Direct similarity lookup mapping used (no multi-hop paths found).")
            
        trace_steps.append("\n**Retrieved RDF Triplets used for grounding:**")
        if triplets:
            for t in triplets[:8]:  # Limit for readability
                trace_steps.append(f"- Subject: `{t['subject']}` | Relation: `\"{t['predicate']}\"` | Object: `{t['object']}`")
        else:
            trace_steps.append("- No explicit triplets retrieved.")
            
        audit_trace = "\n".join(trace_steps)
        
        return answer, audit_trace
