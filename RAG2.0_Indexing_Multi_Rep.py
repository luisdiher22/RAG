# RAG 2.0 Indexing with Multi-Representation Retriever Example
#This is the first example of RAG 2.0 using better indexing 
#Seems extremely handy for long documents, so the search can look up the summary first and then locate 
#The information in the original document


from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.stores import InMemoryByteStore
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever


loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

loader = WebBaseLoader("https://lilianweng.github.io/posts/2024-02-05-human-data-quality/")
docs.extend(loader.load())


llm = ChatOllama(model="gpt-oss:20b", temperature=0)


chain =(
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("You are a helpful assistant that creates concise summaries. Provide a clear, comprehensive summary of the following document in 3-5 paragraphs:\n\n{doc}\n\nSummary:")
    | llm
    | StrOutputParser()
)

summaries = chain.batch(docs, {"max_concurrency": 5})
print("Generated Summaries:")
for summary in summaries:
    print(summary)


#Vectorstore to use to index the child chunks

vectorstore = Chroma(collection_name="summaries",
                     embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

#Storage layer for the parents documents
store = InMemoryByteStore()
id_key = "doc_id"


#Retriever
retriever = MultiVectorRetriever(
    vectorstore =vectorstore,
    byte_store=store,
    id_key=id_key
)
doc_ids = [str(uuid.uuid4()) for _ in docs]

#Docs linked to their summaries
summary_docs = [
    Document(page_content=s , metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

#Add summaries to vectorstore
retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))

#Test retrieval
query = "Memory in agents"
sub_docs = vectorstore.similarity_search(query, k=1)
for d in sub_docs:
    print("Retrieved summary:", d.page_content)