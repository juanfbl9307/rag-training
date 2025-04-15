import os
import logging
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path

# DocLing for document processing
from docling import Document, DocumentProcessor
from docling.extractors import PDFExtractor
from docling.chunkers import TokenChunker, SentenceChunker
from docling.embedders import HuggingFaceEmbedder, OpenAIEmbedder

# Vector DB
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChromaDBManager:
    """Manages interactions with ChromaDB"""

    def __init__(self,
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "pdf_collection",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize ChromaDB connection"""
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Set up embedding function
        if embedding_model.startswith("text-embedding"):
            # Requires OPENAI_API_KEY environment variable
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model_name=embedding_model
            )
        else:
            # Use sentence-transformers
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"ChromaDB initialized with collection '{collection_name}'")

    def add_documents(self,
                      texts: List[str],
                      metadatas: List[Dict[str, Any]],
                      ids: List[str]) -> None:
        """Add documents to ChromaDB collection"""
        try:
            # Add in batches to avoid memory issues with large document sets
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                end_idx = min(i + batch_size, len(texts))
                self.collection.add(
                    documents=texts[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )
                logger.info(f"Added batch {i // batch_size + 1} to ChromaDB ({end_idx - i} chunks)")
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": self.persist_directory
        }


class PDFtoChromaDBPipeline:
    """Main pipeline to process PDFs and load them into ChromaDB using DocLing"""

    def __init__(self,
                 pdf_directory: str,
                 chroma_directory: str = "./chroma_db",
                 collection_name: str = "pdf_collection",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 use_ocr: bool = True,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chunking_strategy: str = "sentence"):

        self.pdf_directory = pdf_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_ocr = use_ocr
        self.chunking_strategy = chunking_strategy

        # Initialize DocLing components
        self.pdf_extractor = PDFExtractor(use_ocr=use_ocr)

        # Choose chunker based on strategy
        if chunking_strategy == "token":
            self.chunker = TokenChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:  # Default to sentence chunking
            self.chunker = SentenceChunker(
                max_chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        # Initialize embedder (not used directly but for completeness)
        if embedding_model.startswith("text-embedding"):
            self.embedder = OpenAIEmbedder(model=embedding_model)
        else:
            self.embedder = HuggingFaceEmbedder(model_name=embedding_model)

        # Initialize document processor
        self.doc_processor = DocumentProcessor(
            extractor=self.pdf_extractor,
            chunker=self.chunker,
            embedder=None  # We'll use ChromaDB's embedding function
        )

        # Initialize ChromaDB manager
        self.db_manager = ChromaDBManager(
            persist_directory=chroma_directory,
            collection_name=collection_name,
            embedding_model=embedding_model
        )

    def process(self) -> None:
        """Process all PDFs in the directory and load them into ChromaDB"""
        pdf_files = [f for f in os.listdir(self.pdf_directory)
                     if f.lower().endswith('.pdf')]

        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_directory}")
            return

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        all_chunks = []
        all_metadatas = []
        all_ids = []

        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_directory, pdf_file)
            logger.info(f"Processing {pdf_file}")

            try:
                # Create a Document object
                doc = Document(source=pdf_path)

                # Extract text
                doc = self.pdf_extractor.extract(doc)

                if not doc.text or len(doc.text.strip()) == 0:
                    logger.warning(f"No text extracted from {pdf_file}, skipping")
                    continue

                # Chunk the document
                doc = self.chunker.chunk(doc)

                logger.info(f"Split {pdf_file} into {len(doc.chunks)} chunks")

                # Prepare chunks for ChromaDB
                for i, chunk in enumerate(doc.chunks):
                    chunk_id = f"{os.path.splitext(pdf_file)[0]}_chunk_{i}"
                    metadata = {
                        "source": pdf_file,
                        "chunk_index": i,
                        "total_chunks": len(doc.chunks),
                        "filename": pdf_file,
                        "file_path": pdf_path
                    }

                    all_chunks.append(chunk.text)
                    all_metadatas.append(metadata)
                    all_ids.append(chunk_id)

            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")

        # Add all documents to ChromaDB
        if all_chunks:
            self.db_manager.add_documents(
                texts=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )

            # Print collection info
            collection_info = self.db_manager.get_collection_info()
            logger.info(f"Finished processing. Collection now contains {collection_info['document_count']} chunks")
        else:
            logger.warning("No text chunks were extracted from any PDF files")


def main():
    parser = argparse.ArgumentParser(description="Process PDF files and load them into ChromaDB using DocLing")
    parser.add_argument("--pdf_dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--chroma_dir", default="./chroma_db", help="Directory to store ChromaDB")
    parser.add_argument("--collection", default="pdf_collection", help="Name of the ChromaDB collection")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Size of text chunks")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Overlap between chunks")
    parser.add_argument("--no_ocr", action="store_true", help="Disable OCR for scanned documents")
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2",
                        help="Embedding model to use (all-MiniLM-L6-v2, text-embedding-ada-002, etc.)")
    parser.add_argument("--chunking_strategy", default="sentence", choices=["sentence", "token"],
                        help="Strategy for chunking text (sentence or token)")

    args = parser.parse_args()

    pipeline = PDFtoChromaDBPipeline(
        pdf_directory=args.pdf_dir,
        chroma_directory=args.chroma_dir,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_ocr=not args.no_ocr,
        embedding_model=args.embedding_model,
        chunking_strategy=args.chunking_strategy
    )

    pipeline.process()


if __name__ == "__main__":
    main()