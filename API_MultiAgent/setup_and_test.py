"""
Complete setup and test script for Multi-Agent RAG System using existing APIs
This script:
1. Uses Vectordb_Creator API to create a collection
2. Uses Vectordb_PDFReader API to process PDFs
3. Uses Vectordb_InfoLoader API to load data into Milvus
4. Tests the Multi-Agent API

IMPORTANT: You need to have these servers running:
- Vectordb_Creator API on port 8001
- Vectordb_PDFReader API on port 8002
- Vectordb_InfoLoader API on port 8003
- MultiAgent API on port 8000
"""
import os
import json
import requests
from pathlib import Path
from pprint import pprint

# Configuration
COLLECTION_NAME = "amazon_seller_sops"
SOPS_FOLDER = "sops"  # Folder containing PDF files

# API endpoints
CREATOR_API = "http://localhost:8001"
PDF_READER_API = "http://localhost:8002"
LOADER_API = "http://localhost:8003"
MULTI_AGENT_API = "http://localhost:8000"

print("\n" + "="*80)
print("MULTI-AGENT RAG SYSTEM - COMPLETE SETUP AND TEST USING YOUR APIS")
print("="*80)

# Step 1: Create collection using Vectordb_Creator API
print("\n[1/4] Creating collection using Vectordb_Creator API...")
try:
    response = requests.post(f"{CREATOR_API}/create_collection/{COLLECTION_NAME}")
    if response.status_code == 200:
        result = response.json()
        print(f" {result['message']}")
    else:
        print(f"  Response: {response.text}")
except requests.exceptions.ConnectionError:
    print(f" Could not connect to Vectordb_Creator API at {CREATOR_API}")
    
    exit(1)
except Exception as e:
    print(f" Error: {e}")
    exit(1)

# Step 2: Process PDFs using Vectordb_PDFReader API
print("\n[2/4] Processing PDFs using Vectordb_PDFReader API...")
sops_path = Path(SOPS_FOLDER)

if not sops_path.exists():
    print(f" Folder '{SOPS_FOLDER}' not found")
    exit(1)

pdf_files = list(sops_path.glob("*.pdf"))
if not pdf_files:
    print(f" No PDF files found in '{SOPS_FOLDER}' folder")
    exit(1)

print(f"Found {len(pdf_files)} PDF file(s):")
for pdf in pdf_files:
    print(f"  - {pdf.name}")

all_data = []

try:
    # Upload PDFs to the PDF Reader API
    files = []
    for pdf_file in pdf_files:
        files.append(('files', (pdf_file.name, open(pdf_file, 'rb'), 'application/pdf')))
    
    print(f"\nUploading {len(files)} PDF(s) to Vectordb_PDFReader API...")
    response = requests.post(f"{PDF_READER_API}/upload_pdfs/", files=files)
    
    # Close file handles
    for _, (_, file_handle, _) in files:
        file_handle.close()
    
    if response.status_code == 200:
        result = response.json()
        texts = result['texts']
        metadatas = result['metadatas']
        vectors = result['vectors']
        
        print(f" Processed {len(texts)} chunks from {len(pdf_files)} PDF(s)")
        
        # Prepare data for loading
        all_data = [
            {
                "text": text,
                "metadata": metadata,
                "embedding": vector
            }
            for text, metadata, vector in zip(texts, metadatas, vectors)
        ]
    else:
        print(f" Error: {response.status_code}")
        print(response.text)
        exit(1)
        
except requests.exceptions.ConnectionError:
    print(f" Could not connect to Vectordb_PDFReader API at {PDF_READER_API}")
    exit(1)
except Exception as e:
    print(f" Error: {e}")
    exit(1)

# Step 3: Load data into Milvus using Vectordb_InfoLoader API
print("\n[3/4] Loading data into Milvus using Vectordb_InfoLoader API...")
try:
    payload = {"data": all_data}
    
    print(f"Sending {len(all_data)} chunks to Vectordb_InfoLoader API...")
    response = requests.post(
        f"{LOADER_API}/load_data/{COLLECTION_NAME}",
        json=payload,
        timeout=120
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f" {result['message']}")
    else:
        print(f" Error: {response.status_code}")
        print(response.text)
        exit(1)
        
except requests.exceptions.ConnectionError:
    print(f" Could not connect to Vectordb_InfoLoader API at {LOADER_API}")
    exit(1)
except Exception as e:
    print(f" Error: {e}")
    exit(1)

# Now test the Multi-Agent API
print("\n" + "="*80)
print("TESTING MULTI-AGENT API")
print("="*80)

print("\n[TEST 1] Creating Multi-Agent Configuration...")
try:
    config = {
        "system_prompt": "You are an Amazon Seller Central expert. Answer questions based on the official SOPs provided in the documents.",
        "collection_name": COLLECTION_NAME
    }
    
    response = requests.post(f"{MULTI_AGENT_API}/create_multi_agent", json=config, timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        
        # Check if there's an error in the response
        if "error" in result:
            print(f"\n API returned an error: {result['error']}")
            if "available_collections" in result:
                print(f"Available collections: {result['available_collections']}")
            exit(1)
        
        session_id = result["session_id"]
        print("\n Multi-Agent System Created!")
        pprint(result)
    else:
        print(f"\n Error: {response.status_code}")
        print(response.text)
        exit(1)
        
except requests.exceptions.ConnectionError:
    print("\n Could not connect to Multi-Agent API")
    exit(1)
except Exception as e:
    print(f"\n Error: {e}")
    exit(1)

# Test querying
print("\n[TEST 2] Querying the Multi-Agent System...")
try:
    test_questions = [
        "What is this document about?",
        "Summarize the main topics covered",
        "What are the key procedures described?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Question {i} ---")
        print(f"Q: {question}")
        
        query = {
            "session_id": session_id,
            "question": question
        }
        
        response = requests.post(f"{MULTI_AGENT_API}/query", json=query, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"A: {result['answer'][:300]}...")
        else:
            print(f" Error: {response.status_code}")
            print(response.text)
            
except Exception as e:
    print(f"\n Error during querying: {e}")
# List sessions
print("\n[TEST 3] Listing Active Sessions...")
try:
    response = requests.get(f"{MULTI_AGENT_API}/sessions")
    if response.status_code == 200:
        result = response.json()
        print("\n Active Sessions:")
        pprint(result)
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*80)
print("SETUP AND TESTING COMPLETED!")
print("="*80)
print(f"\nYou can now use the Multi-Agent API:")
print(f"  Session ID: {session_id}")
print(f"  Collection: {COLLECTION_NAME}")
print(f"  Total documents loaded: {len(all_data)}")
print(f"\nTo query via API:")
print(f'  curl -X POST "{MULTI_AGENT_API}/query" \\')
print(f'    -H "Content-Type: application/json" \\')
print(f'    -d \'{{"session_id": "{session_id}", "question": "Your question here"}}\'')
