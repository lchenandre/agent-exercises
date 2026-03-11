from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

def check_weather(location: str) -> str:
    '''Return the weather forecast for the specified location.'''
    return f"It's always sunny in {location}"

# import os
# DS_API_KEY = os.getenv("DS_API_KEY", default=None)
# print(DS_API_KEY,type(DS_API_KEY))
# os.environ["DEEPSEEK_API_KEY"] = DS_API_KEY
exit
import os
DS_API_KEY = os.getenv("DS_API_KEY", default=None)
llm = ChatOpenAI(
    model="deepseek-chat", 
    openai_api_key=DS_API_KEY, 
    openai_api_base="https://api.deepseek.com/v1", 
)

agent = create_agent(
    # model="deepseek:deepseek-chat",
    model=llm,
    tools=[check_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "what is the weather in sf"}
    ]
})
print(result["messages"][-1].content)
# inputs = {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
# for chunk in graph.stream(inputs, stream_mode="updates"):
#     print(chunk)