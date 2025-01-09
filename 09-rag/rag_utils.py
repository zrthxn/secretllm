import os
import json
import re
from typing import List, Dict
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

def load_markdown_files(directory: str) -> List[Document]:
    """Load all markdown files from a directory and its subdirectories."""
    documents = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    text = f.read()
                    documents.append(Document(page_content=text, metadata={"source": file}))
    return documents

def is_list_item(text: str) -> bool:
    """Check if text is a list item (bulleted or numbered)."""
    return bool(re.match(r'^[\s-]*[-\*\+]|^\s*\d+\.', text))

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into semantically meaningful chunks."""
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # Custom text splitter that preserves semantic structure
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Increased for better context
        chunk_overlap=200,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    split_docs = []
    for doc in documents:
        # First split by headers
        header_splits = markdown_splitter.split_text(doc.page_content)
        
        for header_split in header_splits:
            # Get the header information
            header = (header_split.metadata.get("Header 1", "") or 
                     header_split.metadata.get("Header 2", "") or 
                     header_split.metadata.get("Header 3", ""))
            
            # Split content into lines
            lines = header_split.page_content.split('\n')
            current_chunk = []
            current_length = 0
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Skip empty lines
                if not line:
                    i += 1
                    continue
                
                # Handle key-value pairs
                if ':' in line and not is_list_item(line):
                    current_chunk.append(line)
                    current_length += len(line)
                
                # Handle list items
                elif is_list_item(line):
                    list_items = [line]
                    # Collect all items in the current list
                    while i + 1 < len(lines) and (is_list_item(lines[i + 1].strip()) or not lines[i + 1].strip()):
                        i += 1
                        if lines[i].strip():
                            list_items.append(lines[i].strip())
                    current_chunk.extend(list_items)
                    current_length += sum(len(item) for item in list_items)
                
                # Regular text
                else:
                    current_chunk.append(line)
                    current_length += len(line)
                
                # Create new chunk if size limit reached
                if current_length >= 800:  # Slightly lower than chunk_size to account for overlap
                    split_docs.append(
                        Document(
                            page_content='\n'.join(current_chunk),
                            metadata={
                                "source": doc.metadata["source"],
                                "header": header
                            }
                        )
                    )
                    current_chunk = []
                    current_length = 0
                
                i += 1
            
            # Add remaining content as a chunk
            if current_chunk:
                split_docs.append(
                    Document(
                        page_content='\n'.join(current_chunk),
                        metadata={
                            "source": doc.metadata["source"],
                            "header": header
                        }
                    )
                )
    
    return split_docs

def create_vector_store(documents: List[Document], embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    """Create a FAISS vector store from documents."""
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def load_json_qa(json_path: str) -> List[Document]:
    """Load QA pairs from JSON and convert to documents."""
    with open(json_path, 'r') as f:
        qa_data = json.load(f)
    
    documents = []
    current_topic = None
    current_pairs = []
    
    for item in qa_data:
        if "conversations" in item:
            q = item["conversations"][0]["value"]
            a = item["conversations"][1]["value"]
            
            # Try to identify topic changes based on question content
            topic = q.split()[0] if q.split() else ""
            
            # If topic changed or accumulated enough pairs, create a new document
            if topic != current_topic or len(current_pairs) >= 3:
                if current_pairs:
                    content = "\n\n".join(current_pairs)
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": f"QA_{current_topic}",
                            "type": "qa",
                            "topic": current_topic
                        }
                    ))
                current_pairs = []
                current_topic = topic
            
            current_pairs.append(f"Q: {q}\nA: {a}")
    
    # Add remaining pairs
    if current_pairs:
        content = "\n\n".join(current_pairs)
        documents.append(Document(
            page_content=content,
            metadata={
                "source": f"QA_{current_topic}",
                "type": "qa",
                "topic": current_topic
            }
        ))
    
    return documents

def initialize_rag(markdown_dir: str, json_paths: List[str] = None) -> FAISS:
    """Initialize RAG by loading both markdown and JSON QA documents."""
    # Load markdown documents
    documents = load_markdown_files(markdown_dir)
    split_docs = split_documents(documents)
    
    # Load JSON QA documents if provided
    if json_paths:
        for json_path in json_paths:
            qa_docs = load_json_qa(json_path)
            # No need to split QA docs as they're already in small chunks
            split_docs.extend(qa_docs)
    
    # Create vector store
    vector_store = create_vector_store(split_docs)
    
    return vector_store
