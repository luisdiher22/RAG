#Query structuring with metadata filters using RAG
# We convert user questions into structured database queries
# that include metadata filters to refine search results.
#The test I did was : Question: "videos that are focused on the topic of chat langchain that are published before 2024"
#The result was:
#content_search: SELECT id, title, description, topic, published_date
#FROM videos
#WHERE topic LIKE '%chat langchain%'
#AND published_date < '2024-01-01'
#ORDER BY published_date DESC;
#title_search: Chat LangChain Tutorial Videos (pre‑2024)
#Could definitely work on how its presented but the query structuring seems solid

from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from TutorialSearch import TutorialSearch

system = """You are an expert at converting user questions into database queries.\
        You have access to a database of tutorial videos about a software library.\
        Given a question, return a database query optimized to retrieve the most relevant results.
        
        If there are words you are not familiar with, do not try to rephrase them."""
prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system),
            ("human","{question}"),
        ]
    )

llm = ChatOllama(model="gpt-oss:20b", temperature=0)
structured_llm = llm.with_structured_output(TutorialSearch)
query_analyzer = prompt | structured_llm

result = query_analyzer.invoke({"question": "videos that are focused on the topic of chat langchain that are published before 2024"})
result.pretty_print()