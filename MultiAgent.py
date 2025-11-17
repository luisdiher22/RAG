# I will be doing a multi-agent RAG system , consisting of 3 total agents
# Each agent will have its own RAG chain and will be specialized in a specific area
# Agent 1 will search for results on the rag used to search on the local documents
# Agent 2 will search for results on the web
# Agent 3 will be the final agent that will combine the results of the previous agents and generate the final answer
# We will also have an orchestrator agent that will decide which agent to use based on the question
# I will be using the langgraph library to create the multi-agent system
# I will also be using Milvus as the vector store for the local documents
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
from pymilvus import MilvusClient, DataType, Collection
from langchain.agents import create_agent
from langchain.tools import tool
load_dotenv()


# Connect to Milvus
client = MilvusClient(uri="http://localhost:19530",
                      token="root:Milvus")

# Create database
if "rag_db" not in client.list_databases():
    client.create_database(db_name="rag_db")

# Switch to the db
client.using_database("rag_db")

# Drop collection if exists (for clean start)
if client.has_collection("rag_collection"):
    client.drop_collection("rag_collection")
    print("Dropped existing collection")

# Define collection schema
schema = MilvusClient.create_schema(auto_id=True)

# Define fields
#id field: INT64, primary key, auto_id
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)

#text field: STRING, max_length=65535
schema.add_field("text", DataType.VARCHAR, max_length=65535)

#metadata field: STRING, max_length=65535
schema.add_field("metadata", DataType.VARCHAR, max_length=65535)

#embedding field: FLOAT_VECTOR, dim=768
#384 is the dimension of the embeddings from HuggingFace sentence-transformers/all-MiniLM-L6-v2
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=384)


#Super unnecesary step, but just to show how the table looks like
type_map = {v: k for k, v in DataType.__members__.items()}
print("Final schema:")
for f in schema.to_dict()["fields"]:
    dtype_num = f["type"]
    dtype_name = type_map.get(dtype_num, "UNKNOWN")
    print(f"- {f['name']}: {dtype_name}")

#Next step is to create the collection

# Create collection first (without index)
client.create_collection(
    collection_name="rag_collection",
    schema=schema
)

print("Collection created, now loading documents...")

#Document loading from sops folder - locally saved SOPs as PDFs
# Load all PDF files from sops folder

docs = []
failed_pdfs = []
for pdf_file in Path("sops").glob("*.pdf"):
    try:
        loader = PyPDFLoader(str(pdf_file))
        docs.extend(loader.load())
    except Exception as e:
        failed_pdfs.append((pdf_file.name, str(e)))
        print(f"  Failed to load {pdf_file.name}: {str(e)[:100]}")
        continue

if failed_pdfs:
    print(f"\n  Failed to load {len(failed_pdfs)} PDFs")
    
print(f" Loaded {len(docs)} documents from {len(list(Path('sops').glob('*.pdf'))) - len(failed_pdfs)} PDF files")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1500, chunk_overlap=100)
doc_splits = text_splitter.split_documents(docs)

#Now lets do the embedding and add to milvus
print("Generating embeddings...")
texts = [doc.page_content for doc in doc_splits]
metadatas = [str(doc.metadata) for doc in doc_splits]
vectors = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2").embed_documents(texts)

#Build rows
#Milvus is quite picky about the data format when inserting

rows = [
    {
        "text": t,
        "metadata": m,
        "embedding": v
    }
    for t, m, v in zip(texts, metadatas, vectors)
]

# Inserting data into Milvus
print("Inserting data into Milvus...")
res = client.insert(collection_name="rag_collection",
                            data=rows
                        )
print(f"Inserted {res['insert_count']} entities into Milvus collection 'rag_collection'")

# Now we create the index and load
print("Creating index...")
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="embedding",
    metric_type="COSINE",
    index_type="IVF_FLAT",
    index_name="vector_index",
    params={ "nlist": 128 }
)
client.create_index("rag_collection", index_params=index_params)

print("Loading collection...")
client.load_collection("rag_collection")

#Quick test to see if collection is loaded
res = client.get_load_state(collection_name="rag_collection")
print(f"Load state: {res}")

# Flush to ensure data is persisted
# Flush is almost like an F5 for databases
client.flush("rag_collection")

#Lets check how many entities we have (after flush)
stats = client.get_collection_stats(collection_name="rag_collection")
count = int(stats["row_count"])
print(f" Total entities in collection: {count}")

#That is the whole indexing process done. We got our embeddings into Milvus successfully, running on Docker!

# Now we can build the multi-agent RAG system

# Define the LLM to be used by all agents
llm = ChatOllama(model="gpt-oss:20b", temperature=0)

# Initialize embeddings model (reuse for queries)
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define Tools for agents
#Here is the big first change, we need to define tools that will be used by the agents
#They will use it if their prompt tells them to do so
#For now it will only have one tool, the local sop search
#Another tool in this context that would be useful, is for it to display a specific SOP given its name or id
#But for now we will keep it simple
@tool
def search_local_sops(query: str) -> str:
    """Search internal Amazon Seller Central SOPs for relevant procedures and instructions.
    Use this when the question is about Amazon Seller Central processes, policies, or how-to guides."""
    
    # Generate embedding for the query
    query_vector = embeddings_model.embed_query(query)
    
    # Search in Milvus
    search_results = client.search(
        collection_name="rag_collection",
        data=[query_vector],
        limit=3,  # Top 3 most relevant documents
        output_fields=["text", "metadata"]
    )
    
    if not search_results or not search_results[0]:
        return "No relevant SOPs found in the local database."
    
    # Format results
    context = []
    for i, hit in enumerate(search_results[0], 1):
        context.append(f"Document {i} (score: {hit['distance']:.3f}):\n{hit['entity']['text']}\n")
    
    return "\n".join(context)

#This tool will be used by our second agent, teh web search agent
#In this specific scenario, this agent shouldnt be used often, since most questions should be on the SOPs
@tool
def search_web(query: str) -> str:
    """Search the web for current information, news, or topics not covered in internal SOPs.
    Use this when the question is about general information, current events, or non-SOP topics."""
    
    web_search_tool = TavilySearchResults(k=3)
    results = web_search_tool.invoke(query)
    
    formatted_results = []
    for i, result in enumerate(results, 1):
        formatted_results.append(f"Result {i}:\n{result.get('content', '')}\nSource: {result.get('url', '')}\n")
    
    return "\n".join(formatted_results)

 


# Creating specialized sub-agents


# Agent 1: Local Document Search Agent
# Here is where we are asking him to use the tool
# And since the answers are based on official documentation, we need him to cite them, so the worker using it can verify
# the information if needed
LOCAL_AGENT_PROMPT = """You are an expert assistant specialized in Amazon Seller Central Standard Operating Procedures (SOPs).

Your task is to:
1. Use the search_local_sops tool to find relevant internal documentation
2. Analyze the retrieved SOPs carefully
3. Provide accurate, step-by-step answers based ONLY on the SOP content
4. If the SOPs don't contain the answer, clearly state that

Always cite which SOP or document you're referencing."""

local_agent = create_agent(
    llm,
    tools=[search_local_sops],
    system_prompt=LOCAL_AGENT_PROMPT,
)

# Agent 2: Web Search Agent
# Here we are asking him to use the tool
# And since the answers are based on web results, we need him to cite them, so the worker using it can verify
# the information if needed
WEB_AGENT_PROMPT = """You are a web research assistant.

Your task is to:
1. Use the search_web tool to find current information online
2. Synthesize the web results into a clear answer
3. Cite your sources
4. If the web results don't contain the answer, clearly state that, do not fabricate information.

Use this when internal SOPs are insufficient or the question requires external information."""

web_agent = create_agent(
    llm,
    tools=[search_web],
    system_prompt=WEB_AGENT_PROMPT,
)



# Agent 3: Result Synthesis Agent
# This agent combines results without needing tools - it receives pre-formatted input
SYNTHESIS_AGENT_PROMPT = """You are an expert at combining information from multiple sources.

You will receive:
- Results from internal Amazon Seller Central SOPs (authoritative, always prioritize)
- Results from web search (supplementary, use for context or missing info)

Your task:
1. **Prioritize SOP information** - This is official and should be your primary answer
2. **Enrich with web info** - Only add web details if they provide useful context or the SOPs are incomplete
3. **Create a unified answer** - Combine both seamlessly, don't just list them separately
4. **Cite sources** - Mention which SOP and which web sources you used
5. **If conflict** - Always trust the SOP over web results

Format: Provide a clear, step-by-step answer with citations."""

synthesis_agent = create_agent(
    llm,
    tools=[],  # No tools needed - just combines the text inputs
    system_prompt=SYNTHESIS_AGENT_PROMPT,
)   





# Define the Multi-Agent State
#This is our orchestrator that will decide which agent to call based on the question
class MultiAgentState(TypedDict):
    """State for the multi-agent system"""
    question: str
    local_answer: str
    web_answer: str
    final_answer: str
    next_agent: str

# Router will use LLM directly in router_node function
# No need for a separate routing_agent


# Node functions for the graph

# Router Node
# This node decides which agent(s) to call based on the question
def router_node(state: MultiAgentState) -> MultiAgentState:
    """Routes the question to appropriate agent(s) using LLM decision"""
    print("\n ROUTER analyzing question...")
    
    question = state["question"]
    
    # Ask the LLM router to decide
    router_prompt = f"""Question: {question}

Analyze this question and decide:
- If it's about Amazon Seller Central (SOPs, processes, merchant fulfillment, FBA, inventory, orders, tracking, etc.), respond with EXACTLY: LOCAL
- If it's a general question or about current events/news, respond with EXACTLY: WEB_ONLY

Respond with only one word: LOCAL or WEB_ONLY"""
    
    result = llm.invoke(router_prompt)
    decision = result.content.strip().upper()
    
    print(f"   Router decision: {decision}")
    
    if "LOCAL" in decision:
        return {**state, "next_agent": "local"}
    else:
        return {**state, "next_agent": "web_only"}

# Local Agent Node
# This node calls the local RAG agent
def local_agent_node(state: MultiAgentState) -> MultiAgentState:
    """Calls the local RAG agent"""
    print("\n Calling LOCAL AGENT (searching SOPs)...")
    result = local_agent.invoke(
        {"messages": [{"role": "user", "content": state["question"]}]} 
    )
    local_answer = result['messages'][-1].content
    return {**state, "local_answer": local_answer, "next_agent": "web"}


# Web Agent Node
# This node calls the web search RAG agent
def web_agent_node(state: MultiAgentState) -> MultiAgentState:
    """Calls the web search agent"""
    print("\n Calling WEB AGENT (searching internet)...")
    result = web_agent.invoke(
        {"messages": [{"role": "user", "content": state["question"]}]}
    )
    web_answer = result['messages'][-1].content
    return {**state, "web_answer": web_answer}


# Decision Node after web agent
# This node decides whether to synthesize or finalize
def decide_next_step(state: MultiAgentState) -> str:
    """Decide whether to synthesize (if we have local answer) or end (if web only)"""
    if state.get("local_answer"):
        # We have local answer, so synthesize with web results
        return "synthesis"
    else:
        # No local answer, web is the final answer
        return "end"


# Synthesis Node
# This node calls the synthesis agent to combine results
def synthesis_node(state: MultiAgentState) -> MultiAgentState:
    """Synthesizes results from both agents"""
    print("\n Calling SYNTHESIS AGENT (combining results)...")
    
    synthesis_input = f"""Question: {state["question"]}

Local SOP Answer:
{state.get("local_answer", "Not available")}

Web Search Answer:
{state.get("web_answer", "Not available")}

Combine these into a final answer, prioritizing the SOP information."""
    
    result = synthesis_agent.invoke(
        {"messages": [{"role": "user", "content": synthesis_input}]}
    )
    final_answer = result['messages'][-1].content
    return {**state, "final_answer": final_answer}


# Finalization Node
# This node sets the web answer as final when no synthesis is needed
def finalize_answer(state: MultiAgentState) -> MultiAgentState:
    """Sets web answer as final answer when no synthesis needed"""
    if not state.get("final_answer"):
        # If no final answer yet, use web answer
        return {**state, "final_answer": state["web_answer"]}
    return state

# Build the multi-agent graph
workflow = StateGraph(MultiAgentState)

# Add nodes
workflow.add_node("router", router_node)
workflow.add_node("local", local_agent_node)
workflow.add_node("web", web_agent_node)
workflow.add_node("synthesis", synthesis_node)
workflow.add_node("finalize", finalize_answer)

# Build graph - Define edges
# START -> router (analyze question)
workflow.add_edge(START, "router")

# Conditional routing from router: local or web?
def route_decision(state: MultiAgentState) -> str:
    """Router decides which path to take"""
    return state["next_agent"]

workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "local": "local",      # SOP questions go to local first
        "web_only": "web"      # Non-SOP questions go straight to web
    }
)

# After local, always enrich with web
workflow.add_edge("local", "web")

# After web, decide: synthesize or finalize?
workflow.add_conditional_edges(
    "web",
    decide_next_step,
    {
        "synthesis": "synthesis",  # Had local answer, combine both
        "end": "finalize"          # No local answer, just use web
    }
)

# Both synthesis and finalize lead to END
workflow.add_edge("synthesis", END)
workflow.add_edge("finalize", END)

# Compile the graph
multi_agent_system = workflow.compile()

# Ready to use! Test your questions here:
print("\n" + "="*100)
print("MULTI-AGENT SYSTEM READY")
print("="*100)
print("\nTest your question:")

# Your test question
result = multi_agent_system.invoke({
    "question": "How do I create a removal order?",
    "local_answer": "",
    "web_answer": "",
    "final_answer": "",
    "next_agent": ""
})

print("\n FINAL ANSWER:")
print(result["final_answer"])