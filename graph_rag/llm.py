import logging
import hashlib
import random
import numpy as np
from typing import List, Optional
import google.generativeai as genai
from graph_rag import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("graph_rag.llm")

# Initialize Gemini API
api_available = False
if config.GEMINI_API_KEY:
    try:
        genai.configure(api_key=config.GEMINI_API_KEY)
        api_available = True
        logger.info("Gemini API successfully configured.")
    except Exception as e:
        logger.warning(f"Failed to configure Gemini API: {e}. Fallback to mock mode enabled.")
else:
    logger.info("No GEMINI_API_KEY provided. Operating in offline/mock mode.")

active_embedding_dimension = 3072

def get_mock_embedding(text: str, dimension: int = None) -> List[float]:
    """Generates a deterministic pseudo-random embedding vector based on text content."""
    if dimension is None:
        dimension = active_embedding_dimension
        
    # Use MD5 to get a deterministic seed from the text
    hasher = hashlib.md5(text.encode('utf-8'))
    seed = int(hasher.hexdigest(), 16) % (2**32 - 1)
    
    # Use random with seed to generate a normalized vector
    rng = random.Random(seed)
    vector = [rng.gauss(0, 1) for _ in range(dimension)]
    
    # Normalize to unit length
    norm = sum(x**2 for x in vector) ** 0.5
    if norm > 0:
        vector = [x / norm for x in vector]
    
    return vector

def get_embedding(text: str) -> List[float]:
    """Retrieves text embedding using Gemini API, with a deterministic mock fallback."""
    global active_embedding_dimension
    if not text:
        return [0.0] * active_embedding_dimension
        
    if api_available and not config.USE_MOCK_FALLBACK:
        try:
            # Clean model name check
            model = config.EMBEDDING_MODEL
            response = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document"
            )
            emb = None
            if isinstance(response, dict) and 'embedding' in response:
                emb = response['embedding']
            elif hasattr(response, 'embedding') and response.embedding:
                emb = response.embedding
            elif isinstance(response, list):
                emb = response
            elif 'embedding' in response:
                emb = response['embedding']
                
            if emb:
                # Update dimension dynamically to match API output
                active_embedding_dimension = len(emb)
                return emb
        except Exception as e:
            logger.error(f"Gemini embedding API failed: {e}. Falling back to mock embedding.")
            
    # Mock fallback
    return get_mock_embedding(text, dimension=active_embedding_dimension)

def generate_text(prompt: str, system_instruction: Optional[str] = None) -> str:
    """Generates text using Gemini LLM, with a fallback matching standard queries."""
    if api_available and not config.USE_MOCK_FALLBACK:
        try:
            model_name = config.LLM_MODEL
            
            # Combine system instruction and prompt if set (1.5-flash supports system_instruction parameter)
            kwargs = {}
            if system_instruction:
                kwargs['system_instruction'] = system_instruction
                
            model = genai.GenerativeModel(model_name=model_name, **kwargs)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini text generation failed: {e}. Falling back to mock response.")

    # Mock responses for common workflow tasks
    return get_mock_generation(prompt, system_instruction)

def get_mock_generation(prompt: str, system_instruction: Optional[str] = None) -> str:
    """Generates context-rich mock outputs when offline to simulate Gemini operations."""
    prompt_lower = prompt.lower()
    
    # 1. Extractor: Extracting RDF triplets
    if "extract" in prompt_lower and ("triplets" in prompt_lower or "subject" in prompt_lower):
        # We can scan the prompt for entity names and output a JSON list of triplets
        # Let's write a simple rule-based mock extractor that parses sample documents
        triplets = []
        
        # Check if the prompt has specific corporate text
        if "company x" in prompt_lower:
            triplets.append({"subject": "Company X", "predicate": "partners with", "object": "Supplier B"})
            triplets.append({"subject": "Company X", "predicate": "shares", "object": "Customer Data"})
            triplets.append({"subject": "Customer Data", "predicate": "includes", "object": "PII"})
            triplets.append({"subject": "Company X", "predicate": "must comply with", "object": "Federal Privacy Regulation"})
        if "supplier b" in prompt_lower:
            triplets.append({"subject": "Supplier B", "predicate": "stores", "object": "Customer Data"})
            triplets.append({"subject": "Supplier B", "predicate": "undergoes", "object": "Cybersecurity Audit"})
            triplets.append({"subject": "Cybersecurity Audit", "predicate": "is conducted by", "object": "Third-Party Auditor"})
            triplets.append({"subject": "Supplier B", "predicate": "is located in", "object": "Foreign Jurisdiction"})
        if "federal privacy regulation" in prompt_lower or "regulation z" in prompt_lower:
            triplets.append({"subject": "Federal Privacy Regulation", "predicate": "prohibits sharing with", "object": "Foreign Entities"})
            triplets.append({"subject": "Federal Privacy Regulation", "predicate": "mandates", "object": "Encryption"})
            triplets.append({"subject": "Foreign Entities", "predicate": "lack", "object": "Adequate Safeguards"})

        # If we didn't find specific ones, return generic parsed terms
        if not triplets:
            triplets.append({"subject": "Document", "predicate": "discusses", "object": "Topics"})
            
        import json
        entity_names = list(set([t["subject"] for t in triplets] + [t["object"] for t in triplets]))
        entities_list = []
        for name in entity_names:
            ent_type = "Organization" if "Company" in name or "Supplier" in name or "Entities" in name else ("Regulation" if "Regulation" in name or "Act" in name else "Concept")
            entities_list.append({
                "name": name,
                "type": ent_type,
                "description": f"Standard entity representing {name}."
            })
            
        return json.dumps({
            "triplets": triplets, 
            "entities": entities_list,
            "domains": ["Compliance", "Corporate Partnerships", "Data Security", "Federal Regulations"]
        })
        
    # 2. Mapper: Map query to skeleton
    if "map" in prompt_lower and "skeleton" in prompt_lower:
        import json
        # Route based on query keywords
        categories = ["Compliance", "Corporate Partnerships", "Data Security", "Federal Regulations"]
        matched = []
        if "comply" in prompt_lower or "regulation" in prompt_lower or "rule" in prompt_lower:
            matched.append("Compliance")
            matched.append("Federal Regulations")
        if "company x" in prompt_lower or "supplier" in prompt_lower or "partner" in prompt_lower:
            matched.append("Corporate Partnerships")
            matched.append("Data Security")
        if not matched:
            matched = ["Compliance"]
        return json.dumps({"mapped_domains": matched, "confidence": 0.95})
        
    # 3. Navigator/Synthesizer: Answer query
    # Generate structured answer based on input prompt context
    # Let's extract the retrieved context blocks if present
    context_str = ""
    if "context" in prompt_lower or "triplet" in prompt_lower:
        # Just simulate combining the context
        return (
            "Based on the corporate documents and federal guidelines:\n\n"
            "1. **Company X** shares sensitive Customer Data (which includes PII) with **Supplier B** under their corporate partnership agreement.\n"
            "2. **Supplier B** stores this Customer Data but is located in a **Foreign Jurisdiction** and undergoes audits by a Third-Party Auditor.\n"
            "3. **Federal Privacy Regulations** strictly prohibit the transfer or sharing of Customer Data with **Foreign Entities** that lack adequate safeguards.\n\n"
            "**Conclusion/Compliance Issue**: Yes, there is a severe compliance violation. Company X shares Customer Data with Supplier B, who operates in a foreign jurisdiction, which directly conflicts with the Federal Privacy Regulation's prohibition on sharing protected data with foreign entities."
        )
        
    return "Mock LLM Response: Gemini API offline fallback activated. Processed query successfully."
