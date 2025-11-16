# RAG 2.0 with Decomposition Example
# This example demonstrates a Retrieval-Augmented Generation (RAG) architecture
# that incorporates question decomposition to enhance answer quality.
# Out of the methods explored, decomposition, at least for this example, feels really abstract
# But it seems effective in a case in which a question is extremely complex 
# So far, I feel the ideal Query translation is Fusion + Decomposition for questions deemed complex enough
# This probably will change once I learn about more advanced techniques

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

#Prompt Decomposition
#This prompt is based on decomposition techniques for LLMs, where we break down complex questions
#into simpler sub-questions
template = """You are a helpful assistant that generates multiple sub-questions based on a single input question.

Break down this question into 3 specific sub-questions that focus on different aspects:
{question}

Provide only the questions, one per line, without numbers or additional text:"""
prompt_decomposition = ChatPromptTemplate.from_template(template)


#Generate Decomposition Queries Chain
#Had to do a lot of cleaning here because the output was messy
def parse_queries(output):
    queries = [q.strip() for q in output.split("\n") if q.strip() and not q.strip().isdigit()]
    # Remove numbered prefixes like "1. ", "2. ", etc.
    cleaned_queries = []
    for q in queries:
        # Remove leading numbers, dots, dashes, and quotes
        clean_q = q.lstrip("123456789.- \"'").strip()
        if clean_q and len(clean_q) > 10:  # Only keep substantial questions
            cleaned_queries.append(clean_q)
    return cleaned_queries[:3]  # Limit to 3 queries

generate_queries_decomposition = (
    prompt_decomposition
    | llm
    | StrOutputParser()
    | parse_queries
)


## RETRIEVE ##
#Did a quick print here to see the generated sub-questions
question = "What are the main components of a LLM-powered  agent system?"
questions = generate_queries_decomposition.invoke({"question": question})
print("Decomposition Sub-Questions:", questions)

## PROMPT ##

template = """Here is the question you need to answer:

/n --- /n {question} /n --- /n
Here is any available background question + answer pairs:
/n --- /n {q_a_pairs} /n --- /n
Here is additional context relevant to the question:
/n --- /n {context} /n --- /n

Use the above context and any background Q+A pairs to answer the question as best as you can: \n {question}
"""

decomposition_prompt = ChatPromptTemplate.from_template(template)

#Quick method to format Q&A pairs
def format_qa_pair(question,answer):
    formatted_string = ""
    formatted_string += f"Q: {question}\nA: {answer}\n"
    return formatted_string.strip()




## CHAIN ##
#Here we create the RAG chain using decomposition
#We initialize an empty string to hold all the Q&A pairs generated so far
# for each question in the list of sub-questions
# we create a RAG chain that retrieves context using the retriever
# Then we format the Q&A pair and append it to the accumulated string

q_a_pairs = ""

for q in questions:

    rag_chain = (
        {"context": itemgetter("question") | retriever,
         "question": itemgetter("question"),
         "q_a_pairs": itemgetter("q_a_pairs")}
        | decomposition_prompt
        | llm
        | StrOutputParser())
    
    answer = rag_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})
    q_a_pair = format_qa_pair(q, answer)
    q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair
    print(f"Q: {q}")
    print(f"A: {answer}")
    print("---")

print("\n=== FINAL  Q&A PAIRS ===")
print(q_a_pairs)
        





