# Indian Legal Document Search System

A comprehensive search system for Indian legal documents that compares 4 different similarity methods to find the most effective approach for legal document retrieval.

## 🚀 Features

- **Multiple Similarity Methods**: Cosine Similarity, Euclidean Distance, MMR, and Hybrid Similarity
- **Interactive Web Interface**: Built with Streamlit for easy comparison
- **Document Upload**: Support for PDF, DOCX, and TXT files
- **Performance Metrics**: Precision, Recall, F1-Score, and Diversity Score
- **Side-by-Side Comparison**: View results from all methods simultaneously
- **Legal Domain Optimization**: Hybrid method optimized for legal entities

## 🏗️ Architecture

- **Backend**: FastAPI with sentence-transformers
- **Frontend**: Streamlit
- **ML Models**: all-MiniLM-L6-v2 for embeddings
- **Document Processing**: PyPDF2, python-docx

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/legal-document-search.git
cd legal-document-search
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required models:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

## 🚀 Usage

1. Start the FastAPI server:
```bash
uvicorn main:app --reload
```

2. In a new terminal, start the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and go to `http://localhost:8501`

### Using Docker

```bash
docker-compose up
```

## 🔍 Similarity Methods

### 1. Cosine Similarity
- Standard semantic matching using transformer embeddings
- Best for general semantic similarity

### 2. Euclidean Distance
- Geometric distance in embedding space
- Good for spatial relationships

### 3. Maximum Marginal Relevance (MMR)
- Balances relevance and diversity
- Reduces redundancy in results

### 4. Hybrid Similarity
- Combines semantic similarity with legal entity matching
- Formula: 0.6 × Cosine + 0.4 × Legal_Entity_Match

## 📊 Test Queries

- "Income tax deduction for education"
- "GST rate for textile products"
- "Property registration process"
- "Court fee structure"

## 📈 Performance Analysis

The system provides comprehensive performance analysis including:
- Precision and recall metrics
- F1-score comparisons
- Diversity scores
- Method-specific insights

## 🎯 Use Cases

- Legal research and document retrieval
- Comparative analysis of similarity methods
- Educational purposes in information retrieval
- Legal document management systems

## 📚 Dataset

Includes sample documents from:
- Indian Income Tax Act sections
- GST Act provisions
- Property law documents
- Court fee structures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This system is designed for educational and research purposes. Always consult legal professionals for actual legal advice.

## 🔧 Technical Details

- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Legal Entity Extraction**: Pattern-based approach
- **Hybrid Weight**: 0.6 semantic + 0.4 entity matching
- **MMR Lambda**: 0.7 (balances relevance vs diversity)