from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from data.dummy_tool_metadata import dummy_tool_metadata
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# Define system prompt
SYSTEM_PROMPT = """
You are a weather assistant.

You have access to several weather-related tools.

For each user query, select the single most appropriate tool that can best satisfy the user's request.

All user queries require tool usage.

Return only the selected tool name.
"""

class WeatherInput(BaseModel):
    location: str = Field(description="The city or location for the weather query.")
    query: str = Field(description="The original user weather query.")


def make_dummy_weather_tool(name: str, description: str):
    def dummy_weather_func(location: str, query: str) -> str:
        # MVE阶段只验证工具选择，不关注真实执行结果
        return f"{name} was called for location={location}, query={query}"

    return StructuredTool.from_function(
        func=dummy_weather_func,
        name=name,
        description=description,
        args_schema=WeatherInput,
    )

def build_tools_from_metadata(metadata_list: List[Dict[str, Any]]):
    tools = []
    for item in metadata_list:
        tools.append(
            make_dummy_weather_tool(
                name=item["name"],
                description=item["description"],
            )
        )
    return tools

tools = build_tools_from_metadata(dummy_tool_metadata)

import os
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("DS_API_KEY")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("Please set DEEPSEEK_API_KEY or DS_API_KEY before running this script.")

model = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0,
)

# Set up memory
checkpointer = InMemorySaver()

# Create agent
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=tools,
    checkpointer=checkpointer
)

# Run agent
# `thread_id` is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config
)

print(response["messages"][-1].content)

