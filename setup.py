
from setuptools import setup, find_packages

setup(
    name="legal-document-search",
    version="1.0.0",
    description="Indian Legal Document Search System with Multiple Similarity Methods",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "streamlit==1.28.1",
        "sentence-transformers==2.2.2",
        "scikit-learn==1.3.2",
        "numpy==1.24.3",
        "pandas==2.0.3",
        "python-multipart==0.0.6",
        "PyPDF2==3.0.1",
        "python-docx==0.8.11",
        "nltk==3.8.1",
        "spacy==3.7.2",
        "plotly==5.17.0",
        "requests==2.31.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Legal Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
