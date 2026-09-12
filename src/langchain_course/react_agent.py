from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()  # Load environment variables from .env file

class SearchInput(BaseModel):
    """Schema for the source used by the agent."""

    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for the response returned by the agent."""

    answer: str = Field(description="The agent's response to the user's query")
    sources: List[SearchInput] = Field(
        default_factory=list,
        description="A list of sources used to generate the response"
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def react_agent():

    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the current date, time and weather in Tokyo?")]}
    )

    print(response)


if __name__ == "__main__":
    react_agent()
