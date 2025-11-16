# RAG 2.0 with Multi-Query Retrieval using LangChain
# Goal of multi-query is to improve retrieval by generating multiple queries from the original question
# Almost like a shotgun approach, to have more chances of retrieving relevant documents
# Had better results with multi-query retrieval compared to single-query retrieval
# Using Ollama LLM for both multi-query generation and final RAG answer generation


from operator import itemgetter
from langchain_core.load import loads, dumps
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
#I am using a random web page for demo purposes
#bs_kgwargs helps to filter the HTML content to only the relevant parts
#In this case, we are filtering by post-content, post-title, post-header

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content","post-title","post-header")
        )
    ),
)
chunks = loader.load()

## SPLITTING ##
#Using smaller chunks for better retrieval with multiple queries
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)
splits = text_splitter.split_documents(chunks)

## EMBEDDING ##
# Using HuggingFaceEmbeddings to create embeddings for the chunks
# These embeddings will be used for retrieval later
# In here, we also create the vectorstore and the retriever 

vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
retriever = vectorstore.as_retriever()

## LLM ##
#Using Ollama LLM for both multi-query generation and RAG
#In this example, we are going to use it twice, one for generating multiple queries
#and another for the final RAG answer generation
llm = OllamaLLM(model="gemma3")

## RETRIEVAL ##

#Prompt Multi-Query
#This prompt is used to generate multiple different queries from the original question
template = """ You are an AI language model assistant. Your task is to generate five different versions
of the input question to retrieve relevant documents from a vector daabase. By generating multiple queries,
your goal is to help the user overcome some of the limitations of the distance-based search.
Provide these alternative questions separated by newlines. Original Question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)


#Generate Multi-Queries
#This chain takes the original question, generates multiple queries, and splits them into a list
#prompt_perspectives is the one that creates multiple queries
#Then we send that through the llm to generate the queries
#Then we parse the output into a list of strings using StrOutputParser
#Finally, we split the string into a list using a lambda function
generate_queries = (
    prompt_perspectives
    | llm
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)


#Retrieve Unique Union
# This function takes the list of lists of retrieved chunks from multiple queries
# and returns a unique union of chunks
# Its like mashing together all the retrieved chunks and removing duplicates

def get_unique_union(chunks: list[list]):
    """Unique union of retrieved chunks from multiple queries."""
    # Flatten the list of lists, and convert each chunk to string
    flattened_chunks = [dumps(chunk) for sublist in chunks for chunk in sublist]
    #Get unique chunks using set
    unique_chunks = list(set(flattened_chunks))
    #Return
    return [loads(chunk) for chunk in unique_chunks]
                                                                                  

#Function to print generated queries
# Just a testing function to see the generated queries
# So we can see how the multi-query generation is working
def print_generated_queries(question):
    queries = generate_queries.invoke({"question": question})
    print("Generated Queries:")
    for i, q in enumerate(queries,1):
        print(f"{i}. {q}")

## RETRIEVE ##

question = "What are the main components of a RAG architecture?"

#Print generated queries
print_generated_queries(question)

#Retrieve chunks using multi-queries and get unique union
# We are using the generated queries, then mapping them to the retriever
# This will give us a list of lists of chunks
# and then we pass that to get_unique_union to mash them together and remove duplicates
# The final output is a list of unique chunks retrieved using multiple queries
retrieval_chain = generate_queries | retriever.map() | get_unique_union
docs = retrieval_chain.invoke({"question": question})
print("Unique chunks retrieved:", len(docs))  # Number of unique chunks retrieved

## RAG ##

#Prompt RAG
#This prompt is used to generate the final answer using the retrieved context and the original question
template = """Answer the question based on the context below. 
If you don't know the answer, just say that you don't know.
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# This is our second LLM instance for the final RAG answer generation
llm_rag = OllamaLLM(model="gemma3")

# Final RAG Chain
# This chain takes the first chain output (the retrieved chunks)
# and the original question, then passes them through the RAG prompt and LLM to generate the final answer
# We use itemgetter to extract the question from the input dictionary
# We use StrOutputParser to ensure the output is a string

final_rag_chain = (
    { "context": retrieval_chain,
     "question": itemgetter("question")} 
     |prompt 
     | llm_rag
     | StrOutputParser()
)
result = final_rag_chain.invoke({"question": question})
print("Answer:", result)