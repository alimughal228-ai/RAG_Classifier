"""
Streamlit Dashboard for Document AI System

Professional, colorful, high-level dashboard for document processing,
classification, extraction, search, and question answering.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

from classification.classifier import DocumentClassifier
from config import AppConfig, get_config
from embeddings.generator import EmbeddingGenerator
from embeddings.store import VectorStore
from extraction.extractor import InformationExtractor
from pipelines.document_pipeline import DocumentPipeline
from retrieval.retriever import DocumentRetriever

# Page configuration
st.set_page_config(
    page_title="Document AI Dashboard",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling (dark mode compatible)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Dark mode compatible styles */
    @media (prefers-color-scheme: dark) {
        .main-header {
            background: linear-gradient(90deg, #8b9aff 0%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
    }
    
    /* Streamlit dark mode detection */
    [data-theme="dark"] .main-header {
        background: linear-gradient(90deg, #8b9aff 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stMetric {
        background-color: var(--background-color, #f0f2f6);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    
    [data-theme="dark"] .stMetric {
        background-color: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #8b9aff;
    }
    
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        color: #155724;
    }
    
    [data-theme="dark"] .success-box {
        background-color: rgba(40, 167, 69, 0.2);
        border-left: 4px solid #28a745;
        color: #90ee90;
    }
    
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        color: #0c5460;
    }
    
    [data-theme="dark"] .info-box {
        background-color: rgba(23, 162, 184, 0.2);
        border-left: 4px solid #17a2b8;
        color: #7dd3fc;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        color: #856404;
    }
    
    [data-theme="dark"] .warning-box {
        background-color: rgba(255, 193, 7, 0.2);
        border-left: 4px solid #ffc107;
        color: #ffd700;
    }
    
    /* Ensure text is visible in dark mode */
    [data-theme="dark"] {
        color: #ffffff;
    }
    
    [data-theme="dark"] h1, [data-theme="dark"] h2, [data-theme="dark"] h3 {
        color: #ffffff;
    }
    
    [data-theme="dark"] .stDataFrame {
        background-color: rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_output_data(output_path: str) -> Optional[List[Dict[str, Any]]]:
    """Load output.json data with caching."""
    path = Path(output_path)
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


@st.cache_resource
def load_classifier(config: AppConfig) -> DocumentClassifier:
    """Load classifier with caching."""
    classifier = DocumentClassifier(
        model_name=config.classification.model_name,
        confidence_threshold=config.classification.confidence_threshold,
        fallback_threshold=config.classification.fallback_threshold
    )
    classifier.load_model()
    return classifier


@st.cache_resource
def load_embedding_generator(config: AppConfig) -> EmbeddingGenerator:
    """Load embedding generator with caching."""
    generator = EmbeddingGenerator(
        model_name=config.embedding.model_name,
        device=config.embedding.device,
        cache_dir=config.data_dir / "embeddings_cache"
    )
    generator.load_model()
    return generator


def get_vector_store_stats(store_path: Path) -> Dict[str, Any]:
    """Get vector store statistics."""
    try:
        vector_store = VectorStore(
            store_path=store_path,
            similarity_metric="cosine"
        )
        vector_store.load()
        return vector_store.get_stats()
    except Exception:
        return {"num_vectors": 0, "dimension": 0, "similarity_metric": "cosine"}


def dashboard_home():
    """Home/Dashboard page with overview metrics."""
    st.markdown('<h1 class="main-header">📊 Document AI Dashboard</h1>', unsafe_allow_html=True)
    
    config = get_config()
    
    # Load output data
    output_path = config.output_dir / "output.json"
    data = load_output_data(str(output_path))
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if data:
        total_docs = len(data)
        success_docs = len([d for d in data if d.get("status") == "success"])
        classified_docs = len([d for d in data if d.get("classification")])
        extracted_docs = len([d for d in data if d.get("extracted_data") and any(v is not None for v in d.get("extracted_data", {}).values())])
    else:
        total_docs = 0
        success_docs = 0
        classified_docs = 0
        extracted_docs = 0
    
    # Vector store stats
    store_stats = get_vector_store_stats(config.data_dir / "vector_store")
    indexed_docs = store_stats.get("num_vectors", 0)
    
    with col1:
        st.metric(
            label="📄 Total Documents",
            value=total_docs,
            delta=f"{success_docs} processed" if total_docs > 0 else None
        )
    
    with col2:
        st.metric(
            label="✅ Success Rate",
            value=f"{(success_docs/total_docs*100):.1f}%" if total_docs > 0 else "0%",
            delta=f"{success_docs}/{total_docs}" if total_docs > 0 else None
        )
    
    with col3:
        st.metric(
            label="🏷️ Classified",
            value=classified_docs,
            delta=f"{classified_docs/total_docs*100:.1f}%" if total_docs > 0 else None
        )
    
    with col4:
        st.metric(
            label="🔍 Indexed",
            value=indexed_docs,
            delta="Ready for search" if indexed_docs > 0 else "Not indexed"
        )
    
    st.divider()
    
    if data:
        # Classification Distribution Chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Classification Distribution")
            class_counts = {}
            for doc in data:
                if doc.get("classification"):
                    doc_class = doc["classification"].get("class", "Unknown")
                    class_counts[doc_class] = class_counts.get(doc_class, 0) + 1
            
            if class_counts:
                df_class = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
                fig = px.pie(
                    df_class,
                    values="Count",
                    names="Class",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No classification data available")
        
        with col2:
            st.subheader("📈 Processing Status")
            status_counts = {}
            for doc in data:
                status = doc.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
            
            if status_counts:
                df_status = pd.DataFrame(list(status_counts.items()), columns=["Status", "Count"])
                colors = {"success": "#28a745", "failed": "#dc3545", "partial": "#ffc107", "pending": "#6c757d"}
                df_status["Color"] = df_status["Status"].map(colors)
                
                fig = px.bar(
                    df_status,
                    x="Status",
                    y="Count",
                    color="Status",
                    color_discrete_map=colors,
                    text="Count"
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No status data available")
        
        # Confidence Distribution
        st.subheader("🎯 Classification Confidence Distribution")
        confidences = []
        for doc in data:
            if doc.get("classification") and doc["classification"].get("confidence"):
                confidences.append(doc["classification"]["confidence"])
        
        if confidences:
            df_conf = pd.DataFrame(confidences, columns=["Confidence"])
            fig = px.histogram(
                df_conf,
                x="Confidence",
                nbins=20,
                color_discrete_sequence=["#667eea"],
                labels={"Confidence": "Confidence Score", "count": "Number of Documents"}
            )
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No confidence data available")
        
        # Recent Documents Table
        st.subheader("📋 Recent Documents")
        df_docs = pd.DataFrame([
            {
                "File": doc.get("file_name", "Unknown"),
                "Class": (doc.get("classification") or {}).get("class", "N/A"),
                "Confidence": f"{(doc.get('classification') or {}).get('confidence', 0):.3f}" if doc.get("classification") else "N/A",
                "Status": doc.get("status", "unknown"),
                "Extracted Fields": len([v for v in (doc.get("extracted_data") or {}).values() if v is not None])
            }
            for doc in data[-10:]  # Last 10 documents
        ])
        st.dataframe(df_docs, width='stretch', hide_index=True)
        
        # Failed Documents Section
        failed_docs = [doc for doc in data if doc.get("status") == "failed"]
        if failed_docs:
            st.divider()
            st.subheader("❌ Failed Documents")
            st.warning(f"⚠️ {len(failed_docs)} document(s) failed to process")
            
            for doc in failed_docs:
                with st.expander(f"🔴 {doc.get('file_name', 'Unknown')}"):
                    st.write(f"**File:** {doc.get('file_name', 'Unknown')}")
                    st.write(f"**Error:** {doc.get('error', 'Unknown error')}")
                    
                    # Explain common errors
                    error_msg = doc.get('error', '')
                    if 'EOF marker not found' in error_msg:
                        st.info("💡 **Issue**: PDF file is corrupted or incomplete. The file may be damaged or not a valid PDF.")
                    elif 'No text extracted' in error_msg:
                        st.info("💡 **Issue**: Could not extract text from the document. The file may be image-based or encrypted.")
                    elif 'File not found' in error_msg:
                        st.info("💡 **Issue**: The file path is incorrect or the file was moved/deleted.")
    else:
        st.warning("⚠️ No processed documents found. Please process documents first.")
        st.info("💡 Go to 'Process Documents' tab to start processing.")


def process_documents():
    """Document processing page."""
    st.markdown('<h1 class="main-header">⚙️ Process Documents</h1>', unsafe_allow_html=True)
    
    config = get_config()
    
    with st.form("process_form"):
        st.subheader("📁 Select Input Directory")
        input_dir = st.text_input(
            "Input Directory",
            value=str(Path("Dataset_files").absolute()),
            help="Path to directory containing documents to process"
        )
        
        output_file = st.text_input(
            "Output File",
            value="output.json",
            help="Name of output JSON file"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            process_button = st.form_submit_button("🚀 Process Documents", width='stretch')
        with col2:
            clear_cache = st.form_submit_button("🗑️ Clear Cache", width='stretch')
    
    if clear_cache:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared!")
    
    if process_button:
        input_path = Path(input_dir)
        if not input_path.exists():
            st.error(f"❌ Directory not found: {input_dir}")
        else:
            # Find documents
            supported_extensions = ['.pdf', '.docx', '.doc', '.txt', '.md']
            file_paths = []
            for ext in supported_extensions:
                file_paths.extend(input_path.glob(f"*{ext}"))
            
            if not file_paths:
                st.warning(f"⚠️ No supported documents found in: {input_dir}")
            else:
                st.info(f"📄 Found {len(file_paths)} document(s) to process")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Initialize pipeline
                with st.spinner("Initializing pipeline components..."):
                    classifier = load_classifier(config)
                    extractor = InformationExtractor()
                    embedding_generator = load_embedding_generator(config)
                    
                    pipeline = DocumentPipeline(
                        classifier=classifier,
                        extractor=extractor,
                        embedding_generator=embedding_generator
                    )
                
                # Process documents
                results = []
                for i, file_path in enumerate(file_paths):
                    status_text.text(f"Processing: {file_path.name} ({i+1}/{len(file_paths)})")
                    result = pipeline.process_document(file_path, verbose=False)
                    results.append(result)
                    progress_bar.progress((i + 1) / len(file_paths))
                
                # Save results
                output_path = config.output_dir / output_file
                pipeline.save_results(results, output_path)
                
                # Index documents in vector store
                with st.spinner("Indexing documents for search..."):
                    try:
                        vector_store = VectorStore(
                            store_path=config.data_dir / "vector_store",
                            similarity_metric=config.vector_store.similarity_metric
                        )
                        
                        # Prepare embeddings and metadata
                        embeddings_list = []
                        metadata_list = []
                        texts_dict = {}
                        
                        for result in results:
                            if result.get("status") == "success" and result.get("embedding"):
                                doc_id = result.get("document_id")
                                if doc_id:
                                    # Get embedding vector
                                    embedding_data = result.get("embedding", {})
                                    vector = embedding_data.get("vector", [])
                                    if vector:
                                        embeddings_list.append(vector)
                                        
                                        # Prepare metadata
                                        metadata_list.append({
                                            "document_id": doc_id,
                                            "file_name": result.get("file_name", "Unknown"),
                                            "file_path": result.get("file_path", ""),
                                            "classification": result.get("classification", {}).get("class", "Unknown"),
                                            "confidence": result.get("classification", {}).get("confidence", 0.0)
                                        })
                                        
                                        # Store text for snippet extraction
                                        texts_dict[doc_id] = result.get("text", "")
                        
                        if embeddings_list:
                            import numpy as np
                            embeddings_array = np.array(embeddings_list, dtype=np.float32)
                            vector_store.add_embeddings(
                                embeddings=embeddings_array,
                                metadata=metadata_list,
                                texts=texts_dict
                            )
                            vector_store.save()
                            st.success(f"✅ Indexed {len(embeddings_list)} document(s) in vector store")
                        else:
                            st.warning("⚠️ No valid embeddings to index")
                    except Exception as e:
                        st.warning(f"⚠️ Could not index documents: {e}")
                
                # Clear cache to reload data
                st.cache_data.clear()
                
                st.success(f"✅ Successfully processed {len(results)} document(s)!")
                st.balloons()


def search_documents():
    """Document search page."""
    st.markdown('<h1 class="main-header">🔍 Semantic Search</h1>', unsafe_allow_html=True)
    
    config = get_config()
    
    # Check if index exists
    store_path = config.data_dir / "vector_store"
    store_stats = get_vector_store_stats(store_path)
    
    if store_stats["num_vectors"] == 0:
        st.warning("⚠️ No documents indexed. Please process documents first.")
        st.info("💡 Go to 'Process Documents' tab to index documents.")
        return
    
    st.info(f"📊 Index contains {store_stats['num_vectors']} documents")
    
    # Search form
    with st.form("search_form"):
        query = st.text_input(
            "🔎 Search Query",
            placeholder="Enter your search query...",
            help="Search for documents using semantic similarity"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Top-K Results", min_value=1, max_value=20, value=5)
        with col2:
            search_button = st.form_submit_button("🔍 Search", width='stretch')
    
    if search_button and query:
        with st.spinner("Searching..."):
            try:
                # Initialize retriever
                vector_store = VectorStore(
                    store_path=store_path,
                    similarity_metric=config.vector_store.similarity_metric
                )
                vector_store.load()
                
                embedding_generator = load_embedding_generator(config)
                retriever = DocumentRetriever(
                    vector_store=vector_store,
                    embedding_generator=embedding_generator
                )
                
                # Perform search
                results = retriever.retrieve(query=query, top_k=top_k)
                
                if results:
                    st.success(f"✅ Found {len(results)} result(s)")
                    
                    # Display results
                    for i, result in enumerate(results, 1):
                        with st.expander(f"📄 Result {i}: {result.get('file_name', 'Unknown')} (Score: {result['score']:.4f})"):
                            st.write(f"**Document ID:** {result['document_id']}")
                            if 'file_path' in result:
                                st.write(f"**File Path:** {result['file_path']}")
                            st.write(f"**Similarity Score:** {result['score']:.4f}")
                            st.write("**Snippet:**")
                            st.write(result['snippet'])
                else:
                    st.warning("No results found.")
                    
            except Exception as e:
                st.error(f"❌ Error during search: {e}")


def question_answering():
    """Question answering page."""
    st.markdown('<h1 class="main-header">❓ Question Answering</h1>', unsafe_allow_html=True)
    
    config = get_config()
    
    # Check for GROQ API first (preferred)
    try:
        from qa.groq_answerer import GroqQuestionAnswerer, is_groq_available, get_available_models
        groq_available = is_groq_available()
    except ImportError:
        groq_available = False
        GroqQuestionAnswerer = None
        get_available_models = lambda: []
    
    # Check for local LLM as fallback
    try:
        from qa.answerer import QuestionAnswerer, is_qa_available
        local_qa_available = is_qa_available()
    except ImportError:
        local_qa_available = False
        QuestionAnswerer = None
    
    # If neither is available, show simple message
    if not groq_available and not local_qa_available:
        st.warning("⚠️ Question Answering is not available")
        st.info("""
        Question Answering requires either:
        - GROQ API key (set in `.env` file), or
        - Local LLM setup (optional feature)
        
        **Note**: This is an optional feature. All other features work without it!
        """)
        return
    
    # Check if index exists
    store_path = config.data_dir / "vector_store"
    store_stats = get_vector_store_stats(store_path)
    
    if store_stats["num_vectors"] == 0:
        st.warning("⚠️ No documents indexed. Please process documents first.")
        return
    
    # Show which QA method is available
    if groq_available:
        st.success("✅ GROQ API is available - Using best model for fast inference")
        available_models = get_available_models()
    elif local_qa_available:
        st.info("ℹ️ Using local LLM (GROQ API not configured)")
    else:
        st.warning("⚠️ No QA method available")
        return
    
    # QA form
    with st.form("qa_form"):
        question = st.text_area(
            "❓ Your Question",
            placeholder="Ask a question about your documents...",
            height=100,
            help="Ask a question that will be answered using retrieved documents and AI"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            top_k = st.slider("Documents to Retrieve", min_value=1, max_value=10, value=3)
        with col2:
            if groq_available:
                selected_model = st.selectbox(
                    "Model",
                    options=available_models,
                    index=0,
                    help="Select GROQ model (best model selected by default)"
                )
            else:
                model_path = st.text_input(
                    "Model Path",
                    value="",
                    help="Path to GGUF model file (leave empty for default)"
                )
        with col3:
            max_tokens = st.slider("Max Tokens", min_value=64, max_value=1024, value=512)
        
        qa_button = st.form_submit_button("🤖 Get Answer", width='stretch')
    
    if qa_button and question:
        with st.spinner("Processing question..."):
            try:
                # Retrieve documents
                vector_store = VectorStore(
                    store_path=store_path,
                    similarity_metric=config.vector_store.similarity_metric
                )
                vector_store.load()
                
                embedding_generator = load_embedding_generator(config)
                retriever = DocumentRetriever(
                    vector_store=vector_store,
                    embedding_generator=embedding_generator
                )
                
                retrieved_docs = retriever.retrieve(query=question, top_k=top_k)
                
                if not retrieved_docs:
                    st.warning("No relevant documents found for your question.")
                    return
                
                # Get full text for documents (ensure we have actual document content)
                for doc in retrieved_docs:
                    doc_id = doc.get("document_id")
                    if doc_id:
                        # Try to get full text from vector store
                        full_text = vector_store.get_text(doc_id)
                        if full_text:
                            doc["text"] = full_text
                        # If no full text, ensure snippet is available
                        elif not doc.get("text") and not doc.get("snippet"):
                            # Try to get from output.json as fallback
                            output_path = config.output_dir / "output.json"
                            if output_path.exists():
                                import json
                                with open(output_path, 'r', encoding='utf-8') as f:
                                    output_data = json.load(f)
                                    for item in output_data:
                                        if item.get("document_id") == doc_id:
                                            doc["text"] = item.get("text", "")
                                            break
                
                # Use GROQ if available, otherwise fallback to local LLM
                if groq_available:
                    # Initialize GROQ answerer
                    answerer = GroqQuestionAnswerer(model=selected_model)
                    
                    # Generate answer
                    result = answerer.answer(
                        question=question,
                        retrieved_docs=retrieved_docs,
                        max_tokens=max_tokens
                    )
                elif local_qa_available:
                    # Initialize local LLM answerer
                    answerer = QuestionAnswerer(
                        model_path=model_path if model_path else None,
                        verbose=False
                    )
                    answerer.load_model()
                    
                    # Generate answer
                    result = answerer.answer(
                        question=question,
                        retrieved_docs=retrieved_docs,
                        max_tokens=max_tokens
                    )
                else:
                    st.error("No QA method available")
                    return
                
                # Display answer
                st.success("✅ Answer generated!")
                
                st.markdown("### 💬 Answer")
                st.info(result['answer'])
                
                st.markdown("### 📊 Metadata")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Documents Used", result['context_used'])
                with col2:
                    st.metric("Model", result['model'])
                with col3:
                    st.metric("Tokens Generated", result.get('tokens_generated', 0))
                with col4:
                    provider = result.get('provider', 'local')
                    st.metric("Provider", provider.upper())
                
                if result.get('tokens_total'):
                    st.caption(f"Total tokens: {result['tokens_total']} (Prompt: {result.get('tokens_prompt', 0)}, Completion: {result['tokens_generated']})")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())


def analytics():
    """Analytics and insights page."""
    st.markdown('<h1 class="main-header">📈 Analytics & Insights</h1>', unsafe_allow_html=True)
    
    config = get_config()
    output_path = config.output_dir / "output.json"
    data = load_output_data(str(output_path))
    
    if not data:
        st.warning("⚠️ No data available. Process documents first.")
        return
    
    # Extract data for analysis
    df = pd.DataFrame([
        {
            "file_name": doc.get("file_name", "Unknown"),
            "class": (doc.get("classification") or {}).get("class", "Unknown"),
            "confidence": (doc.get("classification") or {}).get("confidence", 0),
            "status": doc.get("status", "unknown"),
            "has_extraction": len([v for v in (doc.get("extracted_data") or {}).values() if v is not None]) > 0,
            "extracted_fields": len([v for v in (doc.get("extracted_data") or {}).values() if v is not None])
        }
        for doc in data
    ])
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", len(df))
    with col2:
        st.metric("Success Rate", f"{(len(df[df['status'] == 'success']) / len(df) * 100):.1f}%")
    with col3:
        st.metric("Avg Confidence", f"{df['confidence'].mean():.3f}")
    
    st.divider()
    
    # Class distribution over time (if we had timestamps, use file order)
    st.subheader("📊 Classification Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        class_counts = df['class'].value_counts()
        fig = px.bar(
            x=class_counts.index,
            y=class_counts.values,
            color=class_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set3,
            labels={"x": "Document Class", "y": "Count"}
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        fig = px.box(
            df,
            x="class",
            y="confidence",
            color="class",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, width='stretch')
    


def main():
    """Main application entry point."""
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/667eea/ffffff?text=Document+AI", width='stretch')
        
        selected = option_menu(
            menu_title="Navigation",
            options=["Dashboard", "Process Documents", "Search", "Question Answering", "Analytics"],
            icons=["house", "gear", "search", "question-circle", "graph-up"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "5!important", "background-color": "#fafafa"},
                "icon": {"color": "#667eea", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#667eea"},
            }
        )
    
    # Route to appropriate page
    if selected == "Dashboard":
        dashboard_home()
    elif selected == "Process Documents":
        process_documents()
    elif selected == "Search":
        search_documents()
    elif selected == "Question Answering":
        question_answering()
    elif selected == "Analytics":
        analytics()


if __name__ == "__main__":
    main()

