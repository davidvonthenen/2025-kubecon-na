"""
RAG Document Ingestion Pipeline
Processes markdown and PDF documents, generates embeddings using Voyage AI,
and stores them in a vector database (ChromaDB).
"""

import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
import PyPDF2
import markdown
from bs4 import BeautifulSoup
import voyageai
from anthropic import Anthropic


@dataclass
class Document:
    """Represents a processed document chunk."""
    content: str
    metadata: Dict[str, str]
    doc_id: str


class DocumentProcessor:
    """Handles document loading and chunking."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_markdown(self, filepath: Path) -> str:
        """Load and convert markdown to plain text."""
        with open(filepath, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert markdown to HTML then extract text
        html = markdown.markdown(md_content)
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()
    
    def load_pdf(self, filepath: Path) -> str:
        """Extract text from PDF."""
        text = []
        with open(filepath, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text.append(page.extract_text())
        return '\n'.join(text)
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < text_len:
                # Look for sentence endings near the chunk boundary
                sentence_end = text.rfind('. ', start, end)
                if sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def process_file(self, filepath: Path) -> List[Document]:
        """Process a single file and return document chunks."""
        suffix = filepath.suffix.lower()
        
        # Load document
        if suffix == '.md':
            text = self.load_markdown(filepath)
        elif suffix == '.pdf':
            text = self.load_pdf(filepath)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        # Chunk text
        chunks = self.chunk_text(text)
        
        # Create Document objects
        documents = []
        for idx, chunk in enumerate(chunks):
            doc_id = hashlib.sha256(
                f"{filepath.name}_{idx}_{chunk[:100]}".encode()
            ).hexdigest()[:16]
            
            doc = Document(
                content=chunk,
                metadata={
                    'source': str(filepath),
                    'filename': filepath.name,
                    'chunk_index': str(idx),
                    'total_chunks': str(len(chunks)),
                    'file_type': suffix[1:]
                },
                doc_id=doc_id
            )
            documents.append(doc)
        
        return documents


class EmbeddingGenerator:
    """Generates embeddings using Voyage AI."""
    
    def __init__(self, api_key: str, model: str = "voyage-3"):
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        # Voyage AI supports batching
        result = self.client.embed(
            texts=texts,
            model=self.model,
            input_type="document"
        )
        return result.embeddings


class VectorStore:
    """Manages vector storage using ChromaDB."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]]
    ) -> None:
        """Add documents and their embeddings to the vector store."""
        self.collection.add(
            ids=[doc.doc_id for doc in documents],
            embeddings=embeddings,
            documents=[doc.content for doc in documents],
            metadatas=[doc.metadata for doc in documents]
        )
    
    def get_stats(self) -> Dict:
        """Get collection statistics."""
        return {
            'total_documents': self.collection.count(),
            'collection_name': self.collection.name
        }


class RAGIngestionPipeline:
    """Main pipeline for RAG document ingestion."""
    
    def __init__(
        self,
        voyage_api_key: str,
        docs_dir: str = "docs",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        persist_directory: str = "./chroma_db",
        batch_size: int = 100
    ):
        self.docs_dir = Path(docs_dir)
        self.batch_size = batch_size
        
        self.processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.embedding_generator = EmbeddingGenerator(voyage_api_key)
        self.vector_store = VectorStore(persist_directory)
    
    def find_documents(self) -> List[Path]:
        """Find all markdown and PDF files in the docs directory."""
        documents = []
        for pattern in ['*.md', '*.pdf']:
            documents.extend(self.docs_dir.rglob(pattern))
        return sorted(documents)
    
    def ingest(self, show_progress: bool = True) -> Dict:
        """
        Run the complete ingestion pipeline.
        
        Returns:
            Dictionary with ingestion statistics.
        """
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.docs_dir}")
        
        # Find all documents
        filepaths = self.find_documents()
        if not filepaths:
            print(f"No documents found in {self.docs_dir}")
            return {'files_processed': 0, 'chunks_created': 0}
        
        print(f"Found {len(filepaths)} documents to process")
        
        all_documents = []
        files_processed = 0
        
        # Process each file
        for filepath in filepaths:
            try:
                if show_progress:
                    print(f"Processing: {filepath.name}")
                
                docs = self.processor.process_file(filepath)
                all_documents.extend(docs)
                files_processed += 1
                
                if show_progress:
                    print(f"  → Created {len(docs)} chunks")
                
            except Exception as e:
                print(f"Error processing {filepath.name}: {e}")
                continue
        
        # Generate embeddings and store in batches
        print(f"\nGenerating embeddings for {len(all_documents)} chunks...")
        
        for i in range(0, len(all_documents), self.batch_size):
            batch_docs = all_documents[i:i + self.batch_size]
            batch_texts = [doc.content for doc in batch_docs]
            
            if show_progress:
                print(f"Processing batch {i // self.batch_size + 1} "
                      f"({len(batch_docs)} chunks)")
            
            # Generate embeddings
            embeddings = self.embedding_generator.generate_embeddings(batch_texts)
            
            # Store in vector database
            self.vector_store.add_documents(batch_docs, embeddings)
        
        stats = self.vector_store.get_stats()
        stats['files_processed'] = files_processed
        stats['chunks_created'] = len(all_documents)
        
        return stats


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Ingest documents for RAG pipeline'
    )
    parser.add_argument(
        '--docs-dir',
        default='docs',
        help='Directory containing documents (default: docs)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Chunk size in characters (default: 1000)'
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=200,
        help='Chunk overlap in characters (default: 200)'
    )
    parser.add_argument(
        '--db-dir',
        default='./chroma_db',
        help='Vector database directory (default: ./chroma_db)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Embedding batch size (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Get API key from environment
    voyage_api_key = os.getenv('VOYAGE_API_KEY')
    if not voyage_api_key:
        raise ValueError("VOYAGE_API_KEY environment variable not set")
    
    # Create and run pipeline
    pipeline = RAGIngestionPipeline(
        voyage_api_key=voyage_api_key,
        docs_dir=args.docs_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        persist_directory=args.db_dir,
        batch_size=args.batch_size
    )
    
    print("Starting RAG document ingestion pipeline...")
    print("=" * 60)
    
    stats = pipeline.ingest()
    
    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Chunks created: {stats['chunks_created']}")
    print(f"Total documents in collection: {stats['total_documents']}")
    print(f"Collection: {stats['collection_name']}")


if __name__ == '__main__':
    main()