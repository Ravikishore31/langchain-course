from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

load_dotenv()  # Load environment variables from .env file

tavily = TavilyClient()  # Initialize Tavily client

@tool
def search(query: str):

    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search results as a string
    """
    print(f"Searching for: {query}")
    return tavily.search(query=query)  # Use Tavily client to perform the search


llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
)
tools = [search]
agent = create_agent(model=llm, tools=tools)

def react_agent():

    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the current date, time and weather in Tokyo?")]}
    )

    print(response)


if __name__ == "__main__":
    react_agent()
