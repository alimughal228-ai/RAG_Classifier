# 📊 Streamlit Dashboard Guide

## Quick Start

Run the dashboard with:

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Features

### 🏠 Dashboard (Home)
- **Key Metrics**: Total documents, success rate, classified count, indexed documents
- **Visualizations**: 
  - Classification distribution (pie chart)
  - Processing status (bar chart)
  - Confidence distribution (histogram)
- **Recent Documents Table**: Last 10 processed documents

### ⚙️ Process Documents
- Select input directory
- Process all documents through the complete pipeline
- Real-time progress tracking
- Automatic indexing for search

### 🔍 Search
- Semantic search using FAISS index
- Top-K results with similarity scores
- Document snippets and metadata
- Interactive result exploration

### ❓ Question Answering
- Ask questions about your documents
- Uses local LLM (requires llama-cpp-python)
- Retrieves relevant context automatically
- Configurable model and parameters

### 📈 Analytics
- Detailed classification analysis
- Confidence score distributions
- Extraction statistics
- Interactive charts and visualizations

## Color Scheme

The dashboard uses a professional purple gradient theme:
- Primary: `#667eea` (Purple)
- Secondary: `#764ba2` (Deep Purple)
- Accent colors from Plotly's Set3 palette

## Requirements

Install dashboard dependencies:

```bash
pip install streamlit streamlit-option-menu plotly pandas
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Tips

1. **First Time**: Process documents in the "Process Documents" tab
2. **Search**: Requires indexed documents (done automatically after processing)
3. **QA**: Requires llama-cpp-python and a GGUF model file
4. **Performance**: Uses caching for faster repeated operations

