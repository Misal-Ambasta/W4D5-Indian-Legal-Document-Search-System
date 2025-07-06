# Q: 2 - Indian Legal Document Search System

Build a search system for Indian legal documents that compares 4 different similarity methods to find the most effective approach for legal document retrieval.

## Technical Requirements

### 1. Implement at least 4 Similarity Methods

- **Cosine Similarity**: Standard semantic matching
- **Euclidean Distance**: Geometric distance in embedding space
- **MMR**: Reduce redundancy in results
- **Hybrid Similarity**: 0.6×Cosine + 0.4×Legal_Entity_Match

### 2. Test Dataset

- Indian Income Tax Act sections
- GST Act provisions
- Sample court judgments
- Property law documents

### 3. Comparison Framework

- **Precision**: Relevant docs in top 5 results
- **Recall**: Coverage of relevant documents
- **Diversity Score**: Result variety (for MMR evaluation)
- **Side-by-side UI**: Show all 4 method results simultaneously

### 4. Web UI Interface

- Document upload (PDF/Word)
- Text query input
- 4-column results comparison
- Performance metrics dashboard

## Test Queries

- "Income tax deduction for education"
- "GST rate for textile products"
- "Property registration process"
- "Court fee structure"

## Deliverables

Submit Github URL with:

- **Working code**
- **UI**: Web app with comparison
- **Analysis**: Performance report with recommendations