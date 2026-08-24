import streamlit as st
import pandas as pd
import networkx as nx
from pathlib import Path
import tempfile
import traceback

# Import core library
from graph_rag import config
from graph_rag.vector_store import VectorStore
from graph_rag.graph_store import GraphStore
from graph_rag.extractor import DocumentExtractor
from graph_rag.orchestrator import MultiAgentOrchestrator

# Set page config
st.set_page_config(
    page_title="Multi-Agent Graph-RAG Dashboard",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize engines
@st.cache_resource
def get_engines():
    vector_store = VectorStore()
    graph_store = GraphStore()
    
    # Load index files if they exist
    v_path = config.INDEX_DIR / "vector_store.json"
    g_path = config.INDEX_DIR / "graph_store.json"
    
    if v_path.exists():
        vector_store.load(v_path)
    if g_path.exists():
        graph_store.load(g_path)
        
    extractor = DocumentExtractor(vector_store, graph_store)
    orchestrator = MultiAgentOrchestrator(vector_store, graph_store)
    
    return vector_store, graph_store, extractor, orchestrator

vector_store, graph_store, extractor, orchestrator = get_engines()

# Helper to save changes
def save_databases():
    v_path = config.INDEX_DIR / "vector_store.json"
    g_path = config.INDEX_DIR / "graph_store.json"
    vector_store.save(v_path)
    graph_store.save(g_path)

# Title & Description
st.title("🕸️ Multi-Agent Graph-RAG System")
st.markdown(
    "This system implements the **Multi-Agent Graph-RAG for Complex reasoning** outlined in the research paper. "
    "Unlike traditional vector RAG, Graph-RAG maps document metadata and text entities into a **dual-layer hierarchy** "
    "(Knowledge Graph Skeleton + Text-Keyword Bipartite Graph) and utilizes collaborating AI agents to perform multi-hop pathfinding "
    "across distributed documents (e.g. cross-company policies)."
)

# Sidebar Options
st.sidebar.header("⚙️ System Configuration")

mock_mode = st.sidebar.toggle("Use Mock Fallback (Offline Mode)", value=config.USE_MOCK_FALLBACK)
if mock_mode != config.USE_MOCK_FALLBACK:
    config.USE_MOCK_FALLBACK = mock_mode
    st.sidebar.warning(f"Mock fallback toggled: {mock_mode}")

traversal_depth = st.sidebar.slider("Graph Traversal Depth", min_value=1, max_value=4, value=2, 
                                    help="Maximum hops the Navigator Agent will traverse starting from mapped entry nodes.")

st.sidebar.divider()
st.sidebar.subheader("📊 Database Statistics")
st.sidebar.write(f"- **Skeleton Domains:** {len(graph_store.skeleton_graph.nodes())}")
st.sidebar.write(f"- **Bipartite Nodes:** {len(graph_store.bipartite_graph.nodes())}")
st.sidebar.write(f"- **Entity Nodes:** {len(graph_store)}")
st.sidebar.write(f"- **Vector Index Chunks:** {len(vector_store)}")

# Create tabs
tab_query, tab_data, tab_stats = st.tabs(["🔍 Search & Reason", "📥 Ingest Documents", "📈 Graph Exploration"])

# --- Tab 1: Search & Reason ---
with tab_query:
    st.subheader("💡 Ask the Graph-RAG Agent")
    st.write("Enter a regulatory compliance or relation question. E.g.:")
    st.code("Does Company X's arrangement with Supplier B comply with Federal Privacy Regulation Z?")
    
    query_text = st.text_input("Enter Query", "")
    
    if st.button("Run Multi-Agent Query", type="primary") and query_text:
        with st.spinner("Executing cooperating agents pipeline..."):
            try:
                # Execute orchestrator query
                result = orchestrator.query(query_text, traversal_depth=traversal_depth)
                
                # Setup columns
                col_left, col_right = st.columns([3, 2])
                
                with col_left:
                    st.success("### Answer")
                    st.markdown(result["answer"])
                    st.markdown(result["audit_trace"])
                    
                with col_right:
                    st.subheader("🔬 Cooperating Agents Transcript")
                    # Display logs inside a scrollable box or text area
                    logs_str = "\n".join(result["agent_logs"])
                    st.text_area("Agent Thought Processes", logs_str, height=450)
                    
                    st.subheader("📍 Mapped Domains")
                    st.write(result["mapped_domains"])
                    
                # RENDER DYNAMIC GRAPH VISUALIZATION OF SEARCH PATH
                st.divider()
                st.subheader("🗺️ Visual Navigation Path")
                st.write("Below is the Bipartite Graph of Entities and Chunks. Nodes in **red** represent the traversed path taken by **The Navigator**.")
                
                # Generate PyVis visualization
                try:
                    from pyvis.network import Network
                    
                    net = Network(height="400px", width="100%", bgcolor="#f8f9fa", font_color="#1e1e1e", heading="Traversal Path Visualization")
                    
                    visited_set = set(result["visited_entities"])
                    
                    # Add all entity nodes retrieved
                    for node_id in visited_set:
                        n_data = graph_store.bipartite_graph.nodes.get(node_id, {})
                        desc = n_data.get("description", "No description")
                        is_start = node_id in result["entry_entities"]
                        
                        color = "#ff4b4b" if is_start else "#ff8a8a"
                        size = 25 if is_start else 20
                        
                        net.add_node(node_id, label=node_id, title=f"Type: Entity\nDesc: {desc}", color=color, size=size)
                        
                    # Add connected chunks
                    for chunk in result["retrieved_chunks"]:
                        chunk_id = chunk["id"]
                        net.add_node(chunk_id, label=chunk_id[:15]+"...", title=f"Type: Chunk\nSource: {chunk['source']}\nText: {chunk['text'][:200]}...", color="#38b6ff", size=15)
                        
                        # Add links to visited entities mentioned in chunk
                        if graph_store.bipartite_graph.has_node(chunk_id):
                            for neighbor in graph_store.bipartite_graph.neighbors(chunk_id):
                                if neighbor in visited_set:
                                    net.add_edge(chunk_id, neighbor, color="#cccccc")
                                    
                    # Add triplets between visited entities
                    for triplet in result["retrieved_triplets"]:
                        sub, obj, pred = triplet["subject"], triplet["object"], triplet["predicate"]
                        if sub in visited_set and obj in visited_set:
                            net.add_edge(sub, obj, label=pred, color="#ff4b4b", width=2)
                            
                    # Save pyvis to local HTML and display it
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_html:
                        net.save_graph(temp_html.name)
                        with open(temp_html.name, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                            
                    st.components.v1.html(html_content, height=450, scrolling=True)
                except Exception as pyvis_err:
                    st.warning("Could not generate interactive graph (requires PyVis library). Showing textual relationship table instead.")
                    # Fallback to tabular representation
                    df_trips = pd.DataFrame(result["retrieved_triplets"])
                    if not df_trips.empty:
                        st.table(df_trips)
                    else:
                        st.write("No relationships traversed.")
                        
            except Exception as e:
                st.error(f"Error executing query: {e}")
                st.code(traceback.format_exc())

# --- Tab 2: Ingest Documents ---
with tab_data:
    st.subheader("📂 Ingest New Corporate Documents")
    st.markdown(
        "Upload raw documents (JSON format containing keys `document_title` and `content`, or plain TXT files). "
        "The extractor will chunk them, extract RDF triplets using Gemini, and update both the vector store and graph store."
    )
    
    uploaded_files = st.file_uploader("Choose TXT or JSON files", accept_multiple_files=True, type=["txt", "json"])
    
    if st.button("Process & Index Uploaded Files", type="primary") and uploaded_files:
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(temp_dir.name)
        
        with st.spinner("Processing documents through Extraction pipeline..."):
            for uploaded_file in uploaded_files:
                file_dest = temp_path / uploaded_file.name
                with open(file_dest, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                st.write(f"📁 Ingesting `{uploaded_file.name}`...")
                extractor.ingest_document(file_dest)
                
            # Save databases
            save_databases()
            st.success("Processing complete! Databases updated and saved.")
            st.rerun()

    st.divider()
    st.subheader("💡 Seed Sample Data")
    st.write("If you want to reload the pre-configured Company X, Supplier B, and Regulation Z datasets, click below:")
    
    if st.button("Load & Index Pre-configured Sample Corporate Data"):
        with st.spinner("Indexing sample data..."):
            sample_dir = config.DATA_DIR / "sample_data"
            if sample_dir.exists():
                for f in sample_dir.glob("*"):
                    if f.suffix in [".json", ".txt"]:
                        st.write(f"Indexing {f.name}...")
                        extractor.ingest_document(f)
                save_databases()
                st.success("Sample data loaded successfully!")
                st.rerun()
            else:
                st.error(f"Sample data folder not found at {sample_dir}")

# --- Tab 3: Graph Exploration ---
with tab_stats:
    st.subheader("🔍 Explore Knowledge Graph Databases")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🏛️ Knowledge Graph Skeleton (Top-layer Domains)")
        skeleton_nodes = list(graph_store.skeleton_graph.nodes(data=True))
        if skeleton_nodes:
            df_skel_nodes = pd.DataFrame([{
                "Domain ID": n[0],
                "Category": n[1].get("category", ""),
                "Description": n[1].get("description", "")
            } for n in skeleton_nodes])
            st.dataframe(df_skel_nodes, use_container_width=True)
        else:
            st.info("No skeleton domains found.")
            
        st.write("### 🕸️ Skeleton Edges (Domain relations)")
        skeleton_edges = list(graph_store.skeleton_graph.edges(data=True))
        if skeleton_edges:
            df_skel_edges = pd.DataFrame([{
                "Source Domain": e[0],
                "Target Domain": e[1],
                "Relationship": e[2].get("relationship", "")
            } for e in skeleton_edges])
            st.dataframe(df_skel_edges, use_container_width=True)
        else:
            st.info("No skeleton edges found.")

    with col2:
        st.write("### 🏷️ RDF Triplets (Subject -> Predicate -> Object)")
        triplets = graph_store.get_rdf_triplets()
        if triplets:
            df_trips = pd.DataFrame(triplets)
            st.dataframe(df_trips, use_container_width=True)
        else:
            st.info("No RDF Triplets found in database.")
            
        st.write("### 📁 Indexed Text Chunks")
        chunks = [n for n, d in graph_store.bipartite_graph.nodes(data=True) if d.get("type") == "chunk"]
        if chunks:
            df_chunks = pd.DataFrame([{
                "Chunk ID": n,
                "Source File": graph_store.bipartite_graph.nodes[n].get("source_file", ""),
                "Preview": graph_store.bipartite_graph.nodes[n].get("text", "")[:100] + "..."
            } for n in chunks])
            st.dataframe(df_chunks, use_container_width=True)
        else:
            st.info("No chunks indexed.")
