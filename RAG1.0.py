#This is my first attempt at creating a RAG system using LangChain with local LLMs and embeddings.
#The goal is to create a simple RAG system that can answer questions based on a book stored in markdown files.
#We will use Ollama for the LLM and HuggingFace for embeddings, and Chroma as the vector store.
#The book is stored in the 'data' directory in markdown format.
#The code will load the documents, split them into chunks, create embeddings, store them in Chroma,
#and then create a RAG chain to answer questions based on the documents.


#After some research, this RAG is not capable of summarizing the whole book for example,
#because the context length of the LLM is limited and the retrieved documents may not cover the entire content.
#Ill keep this one here as version 1.0 of my RAG experiments.
#Next one will be mainly focused on changing to Qdrant as vector store and allowing the user to input their own questions interactively.

#Areas for improvement:
# - Add error handling for file loading and processing
# - Implement caching for embeddings to avoid recomputation
# - Experiment with different chunk sizes and overlaps
# - Add support for other document formats besides markdown
# - Improve prompt template for better answer quality
# - Add logging for better debugging and monitoring
# - Change the vector store to Qdrant for better scalability
# - Allow the user to input their own questions interactively

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

import shutil
# Clear previous Chroma database if exists
shutil.rmtree('./chroma', ignore_errors=True)
if not shutil.rmtree:
    print("Cleared previous Chroma database.")



#1. Load documents
# This loads all md files from the data directory using the DirectoryLoader method
# We use TextLoader to avoid dependency on the unstructured package
# Then we save it into the 'docs' variable
# The glob pattern '**/*.md' ensures that all markdown files in the directory and its subdirectories are loaded
# The recursive=True parameter means that the loader will look into subdirectories as well
# Since we are only using a single book, recursive loading is not strictly necessary here
loader = DirectoryLoader('data', glob='**/*.md', loader_cls=TextLoader,recursive=True)
docs = loader.load()



#2. Split documents into chunks
# Here we split the documents into smaller chunks for better processing
# We use a chunk size of 1500 characters with an overlap of 200 characters
# Overlap means that the end of one chunk will overlap with the beginning of the next chunk by 200 characters
# Then we save the chunks into the 'chunks' variable
# The separators define how the text is split, prioritizing larger breaks first
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200,separators=["\n\n", "\n", ".", "!", "?"])
chunks = text_splitter.split_documents(docs)


#3. Create embeddings
# Here we create embeddings for the document chunks using Ollama Embeddings
# Embeddings are numerical representations of text that capture semantic meaning
# We use a lightweight sentence-transformers model for good performance
# Then we save the embeddings into the 'embeddings' variable
# Future improvement: switch to OllamaEmbeddings for local embeddings if desired
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



#4. Create vector store
# Here we create a Chroma vector store to store the document embeddings
# Chroma is a vector database that allows for efficient similarity search
# We use the local chroma directory to persist the embeddings
# Chroma is easier to set up than Qdrant and doesn't require a separate server
# Then we save the vector store into the 'vectorstore' variable
# I want to switch to Qdrant later for better scalability
# documents=chunks provides the documents to be embedded and stored
# embedding=embeddings specifies the embedding model to use
# persist_directory="./chroma" specifies where to store the vector database
# Retriever is created from the vector store to enable document retrieval based on similarity search
# kwargs={"k": 8} means we want to retrieve the top 8 most relevant documents for a given query
# How we define "relevance" is based on the similarity of the embeddings
# How similarity of the embeddings is calculated is something handled internally by Chroma, needs more research
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})  # Retrieve top 8 relevant documents

#5. LLM Setup
# A LLM is required to generate answers based on the retrieved documents
# It stands for Large Language Model
# Here we use Ollama with the gemma3 model
# Ollama is a local LLM that can run on your machine without needing an internet connection
# I didnt use OpenAI because i dont have tokens 
# So i did some research and found Ollama to be a good alternative for local LLMs
# Then we save the LLM into the 'llm' variable
llm = Ollama(model="gemma3")  # Using your gemma3 model



#6. Create prompt template
# Here we define a prompt template for the RAG chain
# The prompt includes placeholders for context (retrieved documents) and question
#The prompt instructs the LLM to provide the best possible answer based only on the given context
#At the end, we build the prompt which is equal to the PromptTemplate object
#As parameters, we specify that the template has two input variables: context and question
prompt_template = """
You are an expert at doing summaries.
Using only the provided context, generate the best possible answer to the question.
If the context is incomplete, respond with "Insufficient context to answer the question."

Context:
{context}

Question: {question}
Answer:
"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

#7. RAG Chain
# Here we create a RetrievalQA chain using the LLM and the retriever
# RetrievalQA is the classic way to create a retrieval-based QA system in LangChain
# We set return_source_documents to True to get the source documents along with the answer
# I am pretty sure there are newer methods to create RAG chains in LangChain, but this is what i found for now
# llm=llm specifies the language model to use for generating answers
# chain_type="stuff" indicates the method used to combine retrieved documents (stuffing them into the prompt) More research needed here
# Kwargs here allows us to pass additional parameters to customize the chain, such as the prompt template
# return_source_documents=True ensures we get the documents used to generate the answer, so we can see the sources
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)


#8. Ask a question
# Here we ask a question to the RAG chain and get a response
# We ask "Make a one paragraph summary of the book?"
# We use invoke method and access the result from the returned dictionary
# The for loop prints the first 200 characters of each source document, providing a glimpse of the content used to generate the answer
response = rag_chain.invoke({"query": "Make a one paragraph summary of the book?"})
print("Answer:", response["result"])
print("\nSource Documents:")
for i, doc in enumerate(response["source_documents"]):
    print(f"Document {i+1}: {doc.page_content[:200]}...")
