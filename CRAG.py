#CRAG 
from pathlib import Path
from typing import List, TypedDict
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from pprint import pprint
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph, START

# Load environment variables from .env file
load_dotenv()


#Document loading from sops folder - locally saved SOPs as PDFs

# Load all PDF files from sops folder
docs = []
for pdf_file in Path("sops").glob("*.pdf"):
    loader = PyPDFLoader(str(pdf_file))
    docs.extend(loader.load())

print(f"Loaded {len(docs)} documents from {len(list(Path('sops').glob('*.pdf')))} PDF files")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs)

#Add to vectorDB

vectorstore = Chroma.from_documents(documents=doc_splits,
                                    collection_name="rag_chroma",
                                    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
)
retriever = vectorstore.as_retriever()



#LLM

#Data model

class GradeDocuments(BaseModel):
    """Binary score for relevance of retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question: 'yes' or 'no'"
    )

#LLM with function call
llm = ChatOllama(model="gpt-oss:20b", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

#Prompt
system = """You are an expert at evaluating the relevance of documents to a user's question.\n
        if the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human","Retrieved document: \n\n {document}\n\n  User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
question = "How to create a Small & Light Inventory Report?"  
docs = retriever.invoke(question)

# Check the first (most relevant) document 
print(f"\nChecking most relevant document (doc 0):")
print(f"Preview: {docs[0].page_content[:200]}...")
doc_txt = docs[0].page_content
grade_result = retrieval_grader.invoke({"document": doc_txt, "question": question})
print(f"Grading result: {grade_result}")

#RAG Prompt 
rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."),
        ("human", "Question: {question}\n\nContext: {context}\n\nAnswer:"),
    ]
)

#Post-processor to clean up the output
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

#Chain
rag_chain = rag_prompt | llm | StrOutputParser()

#Run
generation = rag_chain.invoke({"context": format_docs(docs), "question": question})
print("\nGenerated Answer:", generation)

#Question re-writer

#llm for question re-writing
llm_rewriter = ChatOllama(model="gpt-oss:20b", temperature=0)

# Prompt
system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm_rewriter | StrOutputParser()
question_rewriter.invoke({"question": question})

web_search_tool = TavilySearchResults(k=3)


#Graph

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str] 

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval 
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents, "question": question}

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


#Compile graph

workflow = StateGraph(GraphState)

#Define nodes
workflow.add_node("retrieve",retrieve) #Retrieve documents
workflow.add_node("grade_documents",grade_documents) #Grade documents
workflow.add_node("transform_query",transform_query) #Re-write question
workflow.add_node("web_search",web_search) #Web search
workflow.add_node("generate",generate) #Generate answer

#Build graph

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"transform_query": "transform_query",
     "generate": "generate",
     },
)
workflow.add_edge("transform_query", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

#compile

app = workflow.compile()

#Run

inputs = {"question": "How to create a Small & Light Inventory Report?"}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
    pprint("\n---\n")

# Final generation
pprint(value["generation"])



