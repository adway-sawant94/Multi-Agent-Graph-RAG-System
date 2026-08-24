import argparse
import sys
from pathlib import Path
from typing import Tuple

# Reconfigure stdout/stderr to support unicode output (e.g. emojis) on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from graph_rag import config
from graph_rag.vector_store import VectorStore
from graph_rag.graph_store import GraphStore
from graph_rag.extractor import DocumentExtractor
from graph_rag.orchestrator import MultiAgentOrchestrator

def setup_engines() -> Tuple[VectorStore, GraphStore]:
    """Instantiates vector and graph store engines and loads existing indexes."""
    vector_store = VectorStore()
    graph_store = GraphStore()
    
    # Load if exists
    v_path = config.INDEX_DIR / "vector_store.json"
    g_path = config.INDEX_DIR / "graph_store.json"
    
    if v_path.exists():
        vector_store.load(v_path)
    if g_path.exists():
        graph_store.load(g_path)
        
    return vector_store, graph_store

def cmd_index(args):
    """Indexes files in a folder into vector and graph databases."""
    vector_store, graph_store = setup_engines()
    extractor = DocumentExtractor(vector_store, graph_store)
    
    folder = Path(args.dir)
    if not folder.exists() or not folder.is_dir():
        print(f"Error: Target directory '{folder}' does not exist or is not a directory.")
        sys.exit(1)
        
    print(f"Scanning directory: {folder}")
    files = list(folder.glob("*.json")) + list(folder.glob("*.txt"))
    
    if not files:
        print("No .json or .txt files found in directory.")
        return
        
    print(f"Found {len(files)} files to ingest. Starting extraction pipeline...")
    for f in files:
        print(f"\nIngesting: {f.name}")
        extractor.ingest_document(f)
        
    # Save the indices
    print("\nSaving indices to disk...")
    v_path = config.INDEX_DIR / "vector_store.json"
    g_path = config.INDEX_DIR / "graph_store.json"
    vector_store.save(v_path)
    graph_store.save(g_path)
    print(f"Saved Vector Store index: {v_path}")
    print(f"Saved Graph Store index: {g_path}")
    print(f"Successfully indexed all documents! Graph contains {len(graph_store)} entities.")

def cmd_query(args):
    """Queries the orchestrator and displays the reasoning trace and final answer."""
    vector_store, graph_store = setup_engines()
    
    if len(vector_store) == 0 or len(graph_store) == 0:
        print("Error: Indexes are empty. Please run the 'index' command first.")
        sys.exit(1)
        
    orchestrator = MultiAgentOrchestrator(vector_store, graph_store)
    
    print(f"Querying Multi-Agent Graph-RAG (Depth: {args.depth}):")
    print(f"Query: '{args.query}'\n")
    
    result = orchestrator.query(args.query, traversal_depth=args.depth)
    
    if args.verbose:
        print("=" * 60)
        print("🔬 MULTI-AGENT EXECUTION TRANSCRIPT")
        print("=" * 60)
        for log in result["agent_logs"]:
            print(log)
        print("=" * 60 + "\n")
        
    print("=" * 60)
    print("🤖 GENERATED GROUNDED ANSWER")
    print("=" * 60)
    print(result["answer"])
    print("\n" + result["audit_trace"])
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description="Mini-Graph-RAG CLI: Vector & Graph Multi-Agent Information Finder"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Index parser
    index_parser = subparsers.add_parser("index", help="Index documents in a directory")
    index_parser.add_argument(
        "--dir", 
        default=str(config.DATA_DIR / "sample_data"),
        help="Path to folder containing documents to index"
    )
    
    # Query parser
    query_parser = subparsers.add_parser("query", help="Query the Graph-RAG system")
    query_parser.add_argument("query", help="The query string to run")
    query_parser.add_argument(
        "--depth", 
        type=int, 
        default=2, 
        help="Traversal search depth for multi-hop pathfinding"
    )
    query_parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Show agent thinking transcripts"
    )
    
    args = parser.parse_args()
    
    if args.command == "index":
        cmd_index(args)
    elif args.command == "query":
        cmd_query(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    from typing import Tuple # added import in main block
    main()
