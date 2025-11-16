#RAG 2.0 using HyDE (Hypothetical Document Embeddings) for query translation
# HyDE is a technique where we generate hypothetical documents based on the query
# and use those generated documents to perform retrieval.
# This can help in scenarios where the original query might be too vague
# Pretty interesting, almost like the multiple query generation technique but here we generate documents instead of queries


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



#HyDE document generation prompt

template = """ Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""
prompt_hyde = ChatPromptTemplate.from_template(template)


#LLM
llm = OllamaLLM(model="gemma3")



# Chain
generate_docs_for_retrieval = (
    prompt_hyde | llm |StrOutputParser()
) 

#Question
question = "What is task decomposition in large language models?"
result = generate_docs_for_retrieval.invoke({"question": question})

# Retrieve
retrieval_chain = generate_docs_for_retrieval | retriever 
retrieved_docs = retrieval_chain.invoke({"question": question})

# Final RAG prompt

template = """ Answer the question based on the context below.
{context}
Question: {question}
"""
prompt_rag = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    prompt_rag
    | llm
    | StrOutputParser()
)
print(final_rag_chain.invoke({"context": retrieved_docs, "question": question}))