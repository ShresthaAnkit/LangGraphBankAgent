from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from bank_tools import tools
from prompts import BANK_PROMPT
import os
from dotenv import load_dotenv
load_dotenv()

llm = init_chat_model(model="gpt-4o-mini", model_provider="openai", temperature=0)
memory = MemorySaver()
config = {"configurable": {"thread_id": "1"},"recursion_limit": 50}

class MessageClassifier(BaseModel):
    message: str
    message_type: Literal["bank", "general"] = Field(
        ...,
        description="Classify if the message is for `bank` or `general` query"
    )
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None      

def chatbot(state: State):    
    chatbot_llm = llm.with_structured_output(MessageClassifier)
    result = chatbot_llm.invoke([
        {
            "role": "system",
            "content": """You are a general Q&A chatbot. Respond to user queries
                        Classify the user message as either:
                        - 'bank': if it asks for bank related task
                        - 'general': if it is a general conversation
                        """
        },
        *state["messages"]
    ])
    return { "messages": [{ "role": "assistant", "content": result.message }], "message_type": result.message_type }

def open_account(state: State):
    last_message = state["messages"][-1]
    return { "messages": [{ "role": "assistant", "content": f"Opened an account with information:\n{last_message.content}"}] }

tool_node = ToolNode(
    tools=tools,
)


def bank_agent_node(state: dict) -> dict:        
    response = llm.bind_tools(tools).invoke([
        {
            "role": "system",
            "content": BANK_PROMPT
        },
        *state["messages"]
    ])    
    return { "messages": response }


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("bank", bank_agent_node)
graph_builder.add_node("tools",tool_node)
graph_builder.add_node("open_account", open_account)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
        "chatbot",
        lambda state: state.get("message_type"),
        { "bank": "bank", "general": END }
    )
graph_builder.add_conditional_edges("bank", tools_condition, {"tools": "tools","__end__": "open_account"})
graph_builder.add_edge("tools", "bank")

graph = graph_builder.compile(checkpointer=memory)

def run_chatbot():
    state = { "messages": [], "message_type": None }
    while True:
        # print(state["messages"])
        user_input = input("User Message: ")
        EXIT_MESSAGES = ["q", "exit", "Q", "bye"]
        if user_input in EXIT_MESSAGES:
            print("Bye")
            break

        state["messages"] = state.get("messages", []) + [
            { "role": "user", "content": user_input }
        ]

        state = graph.invoke(state,config=config)
        
        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]            
            print(f"Assistant Messsage: {last_message.content}")


run_chatbot()
# print(graph.get_graph().draw_ascii())