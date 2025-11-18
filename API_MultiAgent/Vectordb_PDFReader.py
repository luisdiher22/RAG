#This class will handle the loading of PDF documents and their processing into the vector db
#We will use FastAPI to receive the PDF file  
#We will also do the embedding generation here using HuggingFace transformers
from langchain_community.document_loaders import PyPDFLoader
from fastapi import FastAPI, File, UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import asyncio
import tempfile
import os

# FastAPI app
app = FastAPI()
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Endpoint to upload a single PDF and process it
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Load PDF
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1500, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents)

        texts = [doc.page_content for doc in split_docs]
        metadatas = [str(doc.metadata) for doc in split_docs]
        vectors = embeddings_model.embed_documents(texts)

        #This data can now be sent to the Vectordb_InfoLoader to be loaded into the vector db
        return {"texts": texts, "metadatas": metadatas, "vectors": vectors}
    finally:
        # Clean up temporary file
        os.unlink(temp_path)

#Endpoint to upload multiple PDFs and process them
@app.post("/upload_pdfs/")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    all_texts = []
    all_metadatas = []
    all_vectors = []

    for file in files:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Load PDF
            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1500, chunk_overlap=100)
            split_docs = text_splitter.split_documents(documents)

            texts = [doc.page_content for doc in split_docs]
            metadatas = [str(doc.metadata) for doc in split_docs]
            vectors = await asyncio.to_thread(embeddings_model.embed_documents, texts)

            all_texts.extend(texts)
            all_metadatas.extend(metadatas)
            all_vectors.extend(vectors)
        finally:
            # Clean up temporary file
            os.unlink(temp_path)

    #This data can now be sent to the Vectordb_InfoLoader to be loaded into the vector db
    return {"texts": all_texts, "metadatas": all_metadatas, "vectors": all_vectors}


