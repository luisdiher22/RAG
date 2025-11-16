#This is a second attempt at building a RAG architecture using LangChain  
#Pretty similar to RAG1.0.py but this time we are using a web page as the document source
#And we are using pipeline style chaining for better modularity and clarity


import bs4 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

## INDEXING ##

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content","post-title","post-header")
        )
    ),
)
documents = loader.load()

# Split 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
splits = text_splitter.split_documents(documents)

#Embed 

vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

## Retrieval and generation ##

#Prompt
prompt = ChatPromptTemplate.from_template("""You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:""")

#LLM
llm = OllamaLLM(model="gemma3")

# Post-processor to clean up the output
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# Chain
rag_chain = (
    { "context": retriever | format_docs, "question": RunnablePassthrough() }
    | prompt
    | llm
    | StrOutputParser()
)

#Question

result = rag_chain.invoke("What is Artificial Intelligence?")
print("Answer:", result)