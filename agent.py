#I will use this class to create an agent that can answer basic conversion questions using LangChain and Ollama LLM.
#This is as basic as it gets for an agent setup.
#Just testing how ollama works with agents in LangChain.


from langchain.agents import create_agent
from langchain_ollama import ChatOllama

# Initialize the Ollama LLM with the gemma3 model 
llm = ChatOllama(model="gemma3")


# Create the agent with the LLM
# Here, we define a simple agent that can answer questions about local currencies of countries.
messages = [
    (
        "system",
        "You are a helpful assistant that tells the local currency of a given country.",
    ),
    ("user", "What is the local currency of Costa Rica?"),
]

ai_msg = llm.invoke(messages)
ai_msg 
print(ai_msg.content)  # The local currency of country is x.