#This is a RAG-Fusion implementation using LangChain
#Fusion is pretty interesting, pretty similar to the multiquery retrieval, with an added step of fusing the results
#The fusion step combines the results from multiple queries into a single ranked list of documents
#The Reciprocal Rank Fusion algorithm works like this:
#Each document retrieved by each query is assigned a score based on its rank in the results
#If a chunk appears high in any list, it will score well
#If it appears on multiple lists, its score is even better
#If it shows up low on lists it still gets a score but much lower
#The final scores are summed up and the documents are sorted by their total scores to produce a final ranked list
#The final output is a single list of the most consistently relevant chunks


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

#Prompt RAG-Fusion
#This prompt is used to generate multiple different queries from the original question
template = """ You are a helpful assistant that generates multiple search queries
based on a single input query. \n
Generate multiple search queries related to : {question} \n
Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)


#Generate Multi-Queries
#This chain takes the original question, generates multiple queries, and splits them into a list
#prompt_rag_fusion is the one that creates multiple queries
#Then we send that through the llm to generate the queries
#Then we parse the output into a list of strings using StrOutputParser
#Finally, we split the string into a list using a lambda function
generate_queries = (
    prompt_rag_fusion
    | llm
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

#This is the actual function, it is commented because im using one with debugging so we can see how the scores change

#def reciprocal_rank_fusion(results: list[list], k=60):
#    """Perform Reciprocal Rank Fusion on a list of lists of documents."""
#    
#    #Initialize a dictionary to hold fused scores for each document
#    fused_scores = {}
#
#   #Iterate through each list of ranked documents
#    for docs in results:
#        #Iterate through each document and its rank
#        for rank, doc in enumerate(docs):
#            #Convert the document to a string to use as a key
#            doc_str = dumps(doc)
#            #If the document is not already in the fused scores, initialize its score with 0
#            if doc_str not in fused_scores:
#                fused_scores[doc_str] = 0
#            #Calculate the new score using Reciprocal Rank Fusion formula: 1 / (rank + k)
#            fused_scores[doc_str] += 1 / (rank + k)
#    
#    #Sort the documents by their fused scores in descending order
#    reranked_results = [
#        (loads(doc),score)
#        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
#    ]
#    #Return the reranked results as tuples of (document, score)
#    return reranked_results

def debug_reciprocal_rank_fusion(results: list[list], k=60):
    """
    Same as RRF, but prints a visual breakdown of scoring.
    """
    fused_scores = {}

    print("\n=== RRF DEBUG START ===")
    for query_i, docs in enumerate(results, start=1):
        print(f"\n--- Query {query_i} ---")
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)

            # Initialize score
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0

            rr_score = 1 / (rank + k)

            # Print how the score changes
            print(
                f"Doc: {doc_str[:40]}...  | "
                f"rank={rank:<2} | "
                f"added={rr_score:.6f} | "
                f"total_before={fused_scores[doc_str]:.6f} -> "
                f"total_after={fused_scores[doc_str] + rr_score:.6f}"
            )

            # Apply score
            fused_scores[doc_str] += rr_score

    print("\n=== RRF DEBUG END ===\n")

    # Return normal reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results




## RETRIEVE ##

question = "What is an LLM?"
                                                                                  

retrieval_chain_rag_fusion = generate_queries | retriever.map() | debug_reciprocal_rank_fusion
docs = retrieval_chain_rag_fusion.invoke({"question": question})
len(docs)






## RAG ##

#Prompt RAG
#This prompt is used to generate the final answer using the retrieved context and the original question
template = """Answer the following question based on the context below. 

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


# Final RAG Chain
# This chain takes the first chain output (the retrieved chunks)
# and the original question, then passes them through the RAG prompt and LLM to generate the final answer
# We use itemgetter to extract the question from the input dictionary
# We use StrOutputParser to ensure the output is a string

final_rag_chain = (
    { "context": retrieval_chain_rag_fusion,
     "question": itemgetter("question")} 
     |prompt 
     | llm
     | StrOutputParser()
)
result = final_rag_chain.invoke({"question": question})
print("Answer:", result)