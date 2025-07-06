
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os
import re
import PyPDF2
import docx
from io import BytesIO
import nltk
import spacy

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

app = FastAPI(title="Indian Legal Document Search System")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
documents = []
document_embeddings = None
tfidf_vectorizer = None
tfidf_matrix = None

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    method: str
    results: List[Dict[str, Any]]
    metrics: Dict[str, float]

class ComparisonResult(BaseModel):
    query: str
    methods: List[SearchResult]
    summary: Dict[str, Any]

# Initialize the model
@app.on_event("startup")
async def startup_event():
    global model, documents, document_embeddings, tfidf_vectorizer, tfidf_matrix
    
    # Load sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load sample legal documents
    await load_sample_documents()
    
    # Initialize TF-IDF vectorizer for hybrid method
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    
    if documents:
        # Create embeddings for all documents
        texts = [doc['content'] for doc in documents]
        document_embeddings = model.encode(texts)
        
        # Create TF-IDF matrix
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

async def load_sample_documents():
    """Load sample Indian legal documents"""
    global documents
    
    # Sample Indian legal documents
    sample_docs = [
        {
            "id": 1,
            "title": "Income Tax Act - Section 80C",
            "content": "Deduction in respect of life insurance premia, deferred annuity, provident fund, etc. Any amount paid by an assessee in the previous year as life insurance premia, deferred annuity, provident fund, public provident fund, equity linked savings scheme, unit linked insurance plan, national savings certificate, housing loan principal repayment, tuition fees for children education, fixed deposit for 5 years, infrastructure bonds, etc. shall be allowed as deduction under section 80C up to maximum of Rs. 1,50,000.",
            "category": "Income Tax",
            "act": "Income Tax Act",
            "section": "80C"
        },
        {
            "id": 2,
            "title": "GST Act - Textile Products Rate",
            "content": "Goods and Services Tax on textile products including cotton, wool, silk, synthetic fibers, readymade garments, home textiles, technical textiles. Cotton yarn attracts 5% GST, cotton fabrics attract 5% GST, readymade garments attract 12% GST, branded textiles with retail sale price above Rs. 1000 attract 12% GST, while others attract 5% GST. Textile machinery attracts 18% GST.",
            "category": "GST",
            "act": "GST Act",
            "section": "Schedule"
        },
        {
            "id": 3,
            "title": "Property Registration Process",
            "content": "Registration of property documents under Registration Act 1908. Document execution, payment of stamp duty, registration fee, property valuation, title verification, encumbrance certificate, NOC from society, property tax clearance, power of attorney if required, witness requirements, biometric verification, document scanning and registration. Stamp duty varies from 3% to 10% depending on state. Registration fee is typically 1% of property value.",
            "category": "Property Law",
            "act": "Registration Act",
            "section": "17"
        },
        {
            "id": 4,
            "title": "Court Fee Structure",
            "content": "Court fees payable under Court Fees Act 1870 for various judicial proceedings. Civil suits, criminal cases, appeals, revisions, writs, execution proceedings. Ad valorem fee for suits exceeding Rs. 20,000 ranges from 1% to 7.5% depending on suit value. Fixed fee for criminal cases, matrimonial cases, service matters. Supreme Court fees, High Court fees, District Court fees, Magistrate Court fees. E-filing fees, caveat fees, certified copy fees.",
            "category": "Court Fees",
            "act": "Court Fees Act",
            "section": "1-7"
        },
        {
            "id": 5,
            "title": "Income Tax - Education Deduction Section 80E",
            "content": "Deduction in respect of interest on loan taken for higher education. Interest paid on loan taken from financial institution or approved charitable institution for pursuing higher education for self, spouse, children or student for whom assessee is legal guardian. Deduction available for maximum 8 years from the year of payment of interest or until interest is paid, whichever is earlier. No upper limit on deduction amount.",
            "category": "Income Tax",
            "act": "Income Tax Act",
            "section": "80E"
        },
        {
            "id": 6,
            "title": "GST - Service Tax on Educational Services",
            "content": "GST exemption on educational services provided by educational institutions. Pre-school education, school education, college education, university education, vocational education, skill development courses. Transportation services for students, hostel accommodation, examination fees. Coaching institutes providing education up to higher secondary level are exempt. Professional courses, management courses, technical courses may attract 18% GST.",
            "category": "GST",
            "act": "GST Act",
            "section": "Schedule II"
        },
        {
            "id": 7,
            "title": "Property Law - Ownership Rights",
            "content": "Ownership rights in immovable property under Transfer of Property Act 1882. Absolute ownership, limited ownership, joint ownership, tenancy in common, joint tenancy, life estate, leasehold rights, freehold rights. Right to possess, use, enjoy, transfer, mortgage, lease, gift, will, inherit property. Restrictions on transfer, public policy, perpetuity, rule against perpetuity, restraint on alienation.",
            "category": "Property Law",
            "act": "Transfer of Property Act",
            "section": "5-8"
        },
        {
            "id": 8,
            "title": "Criminal Court Jurisdiction",
            "content": "Criminal court jurisdiction under Criminal Procedure Code. Magistrate court jurisdiction, sessions court jurisdiction, high court jurisdiction, supreme court jurisdiction. Territorial jurisdiction, pecuniary jurisdiction, subject matter jurisdiction. Cognizable offenses, non-cognizable offenses, bailable offenses, non-bailable offenses. Summary trial, regular trial, warrant trial, summons trial.",
            "category": "Criminal Law",
            "act": "Criminal Procedure Code",
            "section": "25-35"
        }
    ]
    
    documents = sample_docs

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(BytesIO(file_bytes))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading DOCX: {str(e)}")

def extract_legal_entities(text: str) -> List[str]:
    """Extract legal entities from text"""
    # Simple pattern matching for legal entities
    patterns = [
        r'Section \d+[A-Z]*',
        r'Article \d+[A-Z]*',
        r'Rule \d+[A-Z]*',
        r'Act \d{4}',
        r'Rs\. [\d,]+',
        r'\d+%',
        r'Chapter [IVX]+',
    ]
    
    entities = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities.extend(matches)
    
    return list(set(entities))

def calculate_legal_entity_similarity(query_entities: List[str], doc_entities: List[str]) -> float:
    """Calculate similarity based on legal entities"""
    if not query_entities or not doc_entities:
        return 0.0
    
    common_entities = set(query_entities) & set(doc_entities)
    total_entities = set(query_entities) | set(doc_entities)
    
    return len(common_entities) / len(total_entities) if total_entities else 0.0

def cosine_similarity_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search using cosine similarity"""
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, document_embeddings)[0]
    
    results = []
    for i, sim in enumerate(similarities):
        results.append({
            "document": documents[i],
            "similarity": float(sim),
            "rank": i + 1
        })
    
    # Sort by similarity and return top_k
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]

def euclidean_distance_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search using Euclidean distance"""
    query_embedding = model.encode([query])
    distances = euclidean_distances(query_embedding, document_embeddings)[0]
    
    results = []
    for i, dist in enumerate(distances):
        # Convert distance to similarity (smaller distance = higher similarity)
        similarity = 1 / (1 + dist)
        results.append({
            "document": documents[i],
            "similarity": float(similarity),
            "rank": i + 1
        })
    
    # Sort by similarity and return top_k
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]

def mmr_search(query: str, top_k: int = 5, lambda_param: float = 0.7) -> List[Dict[str, Any]]:
    """Search using Maximum Marginal Relevance (MMR)"""
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, document_embeddings)[0]
    
    selected_docs = []
    selected_indices = []
    
    for _ in range(min(top_k, len(documents))):
        mmr_scores = []
        
        for i, sim in enumerate(similarities):
            if i in selected_indices:
                mmr_scores.append(-float('inf'))
                continue
            
            # Calculate diversity (minimum similarity to selected documents)
            diversity = 0
            if selected_indices:
                doc_similarities = cosine_similarity(
                    document_embeddings[i:i+1], 
                    document_embeddings[selected_indices]
                )[0]
                diversity = max(doc_similarities)
            
            # MMR formula: λ * similarity - (1-λ) * diversity
            mmr_score = lambda_param * sim - (1 - lambda_param) * diversity
            mmr_scores.append(mmr_score)
        
        # Select document with highest MMR score
        best_idx = np.argmax(mmr_scores)
        selected_indices.append(best_idx)
        selected_docs.append({
            "document": documents[best_idx],
            "similarity": float(similarities[best_idx]),
            "mmr_score": float(mmr_scores[best_idx]),
            "rank": len(selected_docs)
        })
    
    return selected_docs

def hybrid_similarity_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search using hybrid similarity (0.6 * Cosine + 0.4 * Legal Entity Match)"""
    query_embedding = model.encode([query])
    cosine_similarities = cosine_similarity(query_embedding, document_embeddings)[0]
    
    # Extract legal entities from query
    query_entities = extract_legal_entities(query)
    
    results = []
    for i, cos_sim in enumerate(cosine_similarities):
        # Extract legal entities from document
        doc_entities = extract_legal_entities(documents[i]['content'])
        
        # Calculate legal entity similarity
        entity_sim = calculate_legal_entity_similarity(query_entities, doc_entities)
        
        # Hybrid similarity: 0.6 * cosine + 0.4 * entity
        hybrid_sim = 0.6 * cos_sim + 0.4 * entity_sim
        
        results.append({
            "document": documents[i],
            "similarity": float(hybrid_sim),
            "cosine_similarity": float(cos_sim),
            "entity_similarity": float(entity_sim),
            "rank": i + 1
        })
    
    # Sort by hybrid similarity and return top_k
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]

def calculate_precision_recall(results: List[Dict], relevant_categories: List[str]) -> Dict[str, float]:
    """Calculate precision and recall metrics"""
    relevant_retrieved = 0
    total_relevant = sum(1 for doc in documents if doc['category'] in relevant_categories)
    
    for result in results:
        if result['document']['category'] in relevant_categories:
            relevant_retrieved += 1
    
    precision = relevant_retrieved / len(results) if results else 0
    recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    }

def calculate_diversity_score(results: List[Dict]) -> float:
    """Calculate diversity score for MMR evaluation"""
    if len(results) < 2:
        return 0.0
    
    categories = [result['document']['category'] for result in results]
    unique_categories = set(categories)
    
    return len(unique_categories) / len(categories)

@app.post("/upload", response_model=Dict[str, str])
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a legal document"""
    global documents, document_embeddings, tfidf_matrix
    
    try:
        file_bytes = await file.read()
        
        # Extract text based on file type
        if file.filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_bytes)
        elif file.filename.endswith('.docx'):
            text = extract_text_from_docx(file_bytes)
        elif file.filename.endswith('.txt'):
            text = file_bytes.decode('utf-8')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Add document to collection
        new_doc = {
            "id": len(documents) + 1,
            "title": file.filename,
            "content": text,
            "category": "Uploaded",
            "act": "Unknown",
            "section": "Unknown"
        }
        
        documents.append(new_doc)
        
        # Recompute embeddings
        texts = [doc['content'] for doc in documents]
        document_embeddings = model.encode(texts)
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
        
        return {"message": f"Document {file.filename} uploaded successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/search", response_model=ComparisonResult)
async def search_documents(query: SearchQuery):
    """Search documents using all similarity methods"""
    if not documents:
        raise HTTPException(status_code=404, detail="No documents found")
    
    # Determine relevant categories for metrics calculation
    relevant_categories = []
    query_lower = query.query.lower()
    if 'income tax' in query_lower or 'education' in query_lower or 'deduction' in query_lower:
        relevant_categories.append('Income Tax')
    if 'gst' in query_lower or 'textile' in query_lower:
        relevant_categories.append('GST')
    if 'property' in query_lower or 'registration' in query_lower:
        relevant_categories.append('Property Law')
    if 'court' in query_lower or 'fee' in query_lower:
        relevant_categories.extend(['Court Fees', 'Criminal Law'])
    
    # If no specific categories detected, consider all as relevant
    if not relevant_categories:
        relevant_categories = list(set(doc['category'] for doc in documents))
    
    # Perform searches with all methods
    cosine_results = cosine_similarity_search(query.query, query.top_k)
    euclidean_results = euclidean_distance_search(query.query, query.top_k)
    mmr_results = mmr_search(query.query, query.top_k)
    hybrid_results = hybrid_similarity_search(query.query, query.top_k)
    
    # Calculate metrics for each method
    methods = []
    
    # Cosine Similarity
    cosine_metrics = calculate_precision_recall(cosine_results, relevant_categories)
    methods.append(SearchResult(
        method="Cosine Similarity",
        results=cosine_results,
        metrics=cosine_metrics
    ))
    
    # Euclidean Distance
    euclidean_metrics = calculate_precision_recall(euclidean_results, relevant_categories)
    methods.append(SearchResult(
        method="Euclidean Distance",
        results=euclidean_results,
        metrics=euclidean_metrics
    ))
    
    # MMR
    mmr_metrics = calculate_precision_recall(mmr_results, relevant_categories)
    mmr_metrics["diversity_score"] = calculate_diversity_score(mmr_results)
    methods.append(SearchResult(
        method="MMR",
        results=mmr_results,
        metrics=mmr_metrics
    ))
    
    # Hybrid Similarity
    hybrid_metrics = calculate_precision_recall(hybrid_results, relevant_categories)
    methods.append(SearchResult(
        method="Hybrid Similarity",
        results=hybrid_results,
        metrics=hybrid_metrics
    ))
    
    # Summary statistics
    summary = {
        "total_documents": len(documents),
        "relevant_categories": relevant_categories,
        "best_precision": max(method.metrics["precision"] for method in methods),
        "best_recall": max(method.metrics["recall"] for method in methods),
        "best_f1": max(method.metrics["f1_score"] for method in methods),
        "avg_precision": sum(method.metrics["precision"] for method in methods) / len(methods),
        "avg_recall": sum(method.metrics["recall"] for method in methods) / len(methods),
        "avg_f1": sum(method.metrics["f1_score"] for method in methods) / len(methods)
    }
    
    return ComparisonResult(
        query=query.query,
        methods=methods,
        summary=summary
    )

@app.get("/documents")
async def get_documents():
    """Get all documents"""
    return {"documents": documents}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "total_documents": len(documents)}
