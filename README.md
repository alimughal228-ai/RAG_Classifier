# 📄 Document AI System

A comprehensive, fully offline document AI system for processing, classifying, extracting, and searching documents using local models and open-source technologies.

## 🌟 Features

- **📥 Document Ingestion**: Load PDF, DOCX, and TXT files
- **🏷️ Document Classification**: Classify documents into Invoice, Resume, Utility Bill, Other, or Unclassifiable
- **🔍 Information Extraction**: Extract structured data based on document type
- **🧠 Vector Embeddings**: Generate embeddings using SentenceTransformers
- **🔎 Semantic Search**: Fast similarity search using FAISS
- **❓ Question Answering**: Optional QA using GROQ API or local LLM
- **📊 Dashboard**: Professional Streamlit dashboard for visualization and interaction

## 🏗️ Architecture

The system follows clean architecture principles with modular, layered design:

```
.
├── ingestion/          # Document loading and preprocessing
├── classification/     # Document type classification
├── extraction/         # Structured data extraction
├── embeddings/        # Vector embedding generation
├── retrieval/         # Semantic document retrieval
├── pipelines/         # End-to-end pipeline orchestration
├── qa/                # Question answering (optional)
├── config.py          # Central configuration
├── cli.py             # Command-line interface
├── app.py             # Streamlit dashboard
└── requirements.txt   # Python dependencies
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Navigate to Project

```bash
cd path/to/Rag
```

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 3: Optional - Setup Question Answering

**Option A: GROQ API (Recommended - Fast & Easy)**

1. Install GROQ package:
   ```bash
   pip install groq
   ```

2. Get API key from [GROQ Console](https://console.groq.com/)

3. Create `.env` file in project root:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

**Option B: Local LLM (Fully Offline)**

1. Install llama-cpp-python:
   ```bash
   pip install llama-cpp-python
   ```

2. Download a GGUF model (e.g., Mistral 7B) from [HuggingFace](https://huggingface.co/TheBloke)

3. Place model in `./models/` directory

## 🚀 Running the Program

### Method 1: Streamlit Dashboard (Recommended)

Launch the interactive dashboard:

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

**Dashboard Features:**
- 📊 Overview metrics and visualizations
- ⚙️ Process documents through complete pipeline
- 🔍 Semantic search interface
- ❓ Question answering interface
- 📈 Analytics and insights

### Method 2: Command-Line Interface

#### Process Documents (Complete Pipeline)

```bash
# Process all documents in a directory
python cli.py process --input_dir ./Dataset_files

# Custom output file
python cli.py process --input_dir ./Dataset_files --output results.json
```

#### Classify Documents

```bash
# Classify documents
python cli.py classify Dataset_files

# Save results to JSON
python cli.py classify Dataset_files --output classifications.json
```

#### Extract Information

```bash
# Extract structured data
python cli.py extract Dataset_files

# Save results
python cli.py extract Dataset_files --output extractions.json
```

#### Generate Embeddings

```bash
# Generate embeddings
python cli.py embed Dataset_files

# Save to file
python cli.py embed Dataset_files --output embeddings.json
```

#### Index Documents

```bash
# Index documents for search
python cli.py index Dataset_files

# Custom store path
python cli.py index Dataset_files --store-path ./my_index
```

#### Search Documents

```bash
# Semantic search
python cli.py search --query "invoice payment terms" --top-k 5

# Custom store path
python cli.py search --query "resume skills" --store-path ./my_index --top-k 10
```

#### Question Answering

```bash
# Ask questions (uses GROQ if available, otherwise local LLM)
python cli.py qa --question "What is the total amount on invoice_1?" --top-k 3

# With custom model path (for local LLM)
python cli.py qa --question "What are the payment terms?" --model-path ./models/mistral.gguf
```

## 📚 Libraries and Methods Used

### Core Dependencies

#### Document Processing
- **PyPDF2** (≥3.0.0): PDF text extraction
- **python-docx** (≥1.1.0): DOCX document parsing

#### Machine Learning & NLP
- **sentence-transformers** (≥2.2.0): 
  - Model: `all-MiniLM-L6-v2` (384-dimensional embeddings)
  - Used for: Document classification and embedding generation
- **torch** (≥2.0.0): PyTorch backend for sentence-transformers
- **scikit-learn** (≥1.3.0): Utility functions for ML operations
- **numpy** (≥1.24.0): Numerical operations and array handling

#### Vector Search
- **faiss-cpu** (≥1.7.4): 
  - Fast similarity search using FAISS (Facebook AI Similarity Search)
  - Index type: Flat index with cosine similarity
  - Used for: Efficient document retrieval

#### Question Answering (Optional)
- **groq** (≥0.4.0): 
  - GROQ API client for fast inference
  - Models: Mixtral 8x7B, Llama 3 (70B/8B)
  - Used for: Cloud-based question answering
- **llama-cpp-python** (optional): 
  - Local LLM inference via LLaMA.cpp
  - Used for: Fully offline question answering

#### Dashboard
- **streamlit** (≥1.28.0): Web dashboard framework
- **streamlit-option-menu** (≥0.3.6): Navigation menu component
- **plotly** (≥5.17.0): Interactive visualizations
- **pandas** (≥2.0.0): Data manipulation and analysis

#### Utilities
- **python-dotenv** (≥1.0.0): Environment variable management
- **tqdm** (≥4.65.0): Progress bars
- **typing-extensions** (≥4.5.0): Enhanced type hints

### Methods and Techniques

#### 1. Document Classification
- **Primary Method**: SentenceTransformers embeddings + cosine similarity
  - Uses pre-trained `all-MiniLM-L6-v2` model
  - Class prototypes: Short reference texts for each category
  - Cosine similarity between document and class embeddings
- **Fallback Method**: Keyword-based heuristics
  - Regex pattern matching for document-specific keywords
  - Confidence thresholding (0.5 for ML, 0.3 for keywords)
- **Output**: Classification with confidence score

#### 2. Information Extraction
- **Rule-based Extraction**: Regex patterns for structured data
- **Date Normalization**: Multiple format support (ISO, MM/DD/YYYY, etc.)
- **Currency Extraction**: Pattern matching for monetary values
- **Type-specific Extractors**:
  - Invoice: invoice_number, date, company, total_amount
  - Resume: name, email, phone, experience_years
  - Utility Bill: account_number, date, usage_kwh, amount_due

#### 3. Embedding Generation
- **Model**: SentenceTransformers `all-MiniLM-L6-v2`
- **Dimension**: 384
- **Normalization**: L2 normalization for cosine similarity
- **Caching**: Disk-based caching for deterministic results
- **Document IDs**: SHA256 hash-based for unique identification

#### 4. Vector Search
- **Index**: FAISS Flat Index
- **Similarity Metric**: Cosine similarity (normalized vectors)
- **Search Method**: Exact nearest neighbor search
- **Storage**: Local disk-based with metadata persistence

#### 5. Question Answering
- **GROQ API** (Primary):
  - Models: Mixtral 8x7B (32K context), Llama 3 70B/8B
  - Fast inference with high-quality responses
  - Context window: Up to 32K tokens (Mixtral)
- **Local LLM** (Fallback):
  - LLaMA.cpp backend
  - GGUF model format
  - Fully offline operation

## 📋 Project Structure

```
Rag/
├── ingestion/
│   ├── __init__.py
│   ├── loader.py          # PDF, DOCX, TXT loaders
│   └── preprocessor.py    # Text cleaning and chunking
├── classification/
│   ├── __init__.py
│   └── classifier.py     # Document classification
├── extraction/
│   ├── __init__.py
│   └── extractor.py      # Structured data extraction
├── embeddings/
│   ├── __init__.py
│   ├── generator.py       # Embedding generation
│   └── store.py          # FAISS vector store
├── retrieval/
│   ├── __init__.py
│   └── retriever.py      # Semantic search
├── pipelines/
│   ├── __init__.py
│   ├── document_pipeline.py  # Complete processing pipeline
│   ├── ingestion_pipeline.py
│   └── rag_pipeline.py
├── qa/                    # Optional question answering
│   ├── __init__.py
│   ├── answerer.py       # Local LLM QA
│   └── groq_answerer.py # GROQ API QA
├── config.py             # Configuration management
├── cli.py                # Command-line interface
├── app.py                # Streamlit dashboard
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🔧 Configuration

Configuration is managed through `config.py`. Key settings:

- **Classification**: Model name, confidence thresholds
- **Embedding**: Model name, device (CPU/CUDA), batch size
- **Vector Store**: Store path, similarity metric
- **Retrieval**: Top-K, similarity threshold

## 📊 Output Format

### Assessment Output (`output.json`)

The main output file follows the assessment specification format:

```json
{
  "invoice_1.pdf": {
    "class": "Invoice",
    "invoice_number": "1001",
    "date": "2025-06-16",
    "company": "Pioneer Ltd",
    "total_amount": 2073.0
  },
  "resume_1.pdf": {
    "class": "Resume",
    "name": "John Doe",
    "email": "john.doe@example.com",
    "phone": "+1-555-799-6125",
    "experience_years": 5
  },
  "utilitybill_1.pdf": {
    "class": "Utility Bill",
    "account_number": "ACC-12345",
    "date": "2025-01-15",
    "usage_kwh": 1250.5,
    "amount_due": 150.75
  }
}
```

### Full Processing Results (`output_full.json`)

A detailed output file with complete metadata is also generated:

```json
[
  {
    "document_id": "abc123...",
    "file_path": "path/to/document.pdf",
    "file_name": "document.pdf",
    "text": "Extracted text content...",
    "classification": {
      "class": "Invoice",
      "confidence": 0.85
    },
    "extracted_data": {
      "invoice_number": "INV-001",
      "date": "2025-01-15",
      "company": "Example Corp",
      "total_amount": 1500.00
    },
    "embedding": {
      "dimension": 384,
      "vector": [0.123, -0.456, ...]
    },
    "status": "success",
    "error": null
  }
]
```

### Search Results

```json
[
  {
    "document_id": "abc123...",
    "score": 0.9234,
    "snippet": "Relevant text snippet...",
    "file_name": "document.pdf"
  }
]
```

## 🎯 Use Cases

1. **Document Management**: Automatically classify and organize documents
2. **Data Extraction**: Extract structured information from unstructured documents
3. **Semantic Search**: Find documents by meaning, not just keywords
4. **Question Answering**: Ask questions about your document collection
5. **Analytics**: Analyze document types, confidence scores, and extraction rates

## 🔒 Privacy & Security

- **Fully Offline**: Core features work without internet connection
- **Local Processing**: All document processing happens locally
- **No Data Sharing**: Documents never leave your machine
- **Optional Cloud**: GROQ API is optional and clearly indicated

## 🐛 Troubleshooting

### Common Issues

**Issue**: "No module named 'sentence_transformers'"
- **Solution**: `pip install sentence-transformers torch`

**Issue**: "FAISS not found"
- **Solution**: `pip install faiss-cpu`

**Issue**: PDF reading errors
- **Solution**: PDF may be corrupted. Check file integrity.

**Issue**: GROQ API not working
- **Solution**: Verify `GROQ_API_KEY` is set in `.env` file

## 📝 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines if applicable]

## 📧 Contact

[Add contact information if needed]

---

**Built with**: Python, SentenceTransformers, FAISS, Streamlit, GROQ API
