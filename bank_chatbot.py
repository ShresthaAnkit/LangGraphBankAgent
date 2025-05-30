from dotenv import load_dotenv
from utils import validate_nepali_phone_number
import os
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from langgraph.types import Command, interrupt
from langchain.agents import AgentExecutor
from langchain.tools import Tool


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
    last_message = state["messages"][-1]
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
@tool
def getAccountOpeningRequiredFields():
    """Get all the fields required for account opening"""
    # print("Account Opening Required Fields Tool")
    print("INFO - Called tool getAccountOpeningRequiredFields")
    return ["name", "phone_number"]
@tool 
def getLoanRequiredFields():
    """Get all the fields required to take a loan"""
    print("INFO - Called tool getLoanRequiredFields")
    return ["name", "loan_amount"]
@tool
def human_response(query) -> str:    
    """Request the missing data from the human"""    
    print("INFO - Called tool human_response")
    # print("Human Response Tool")
    human_response = input(query + ": ")
    return human_response
@tool
def validate_phone_number(phone_number: str) -> bool:
    """Takes a phone number and checks if it is a valid Nepali phone number"""
    print("INFO - Called tool validate_phone_number")
    if not validate_nepali_phone_number(phone_number):
        return False
    print("INFO - Phone number is valid")
    return True
@tool
def validate_name(name: str) -> bool:
    """Takes a name and checks if it is valid. Valid names are at least 2 characters long"""
    print("INFO - Called tool validate_name")
    if len(name) < 2 or not name.isalpha():
        return False
    print("INFO - Name is valid")
    return True

tools = [
    Tool(
        name="get_account_opening_required_fields",
        func=getAccountOpeningRequiredFields,
        description="See what fields need to be request from the user to open an account",
    ),
    Tool(
        name="get_loan_required_fields",
        func=getLoanRequiredFields,
        description="See what fields need to be request from the user to take a loan",
    ),
    Tool(
        name="validate_phone_number",
        func=validate_phone_number,
        description="Check if the phone number is valid",
    ),
    Tool(
        name="validate_name",
        func=validate_name,
        description="Check if the name is valid",
    ),
    Tool(
        name="human_response",
        func=human_response,
        description="Request the missing data from the human",
    )
]
tool_node = ToolNode(
    tools=tools,
)
BANK_PROMPT = """
You are a bank agent that requests user for all the required fields for the required task.
Keep requesting until all the fields are filled.
Do not stop calling tools until all the fields are filled.
Validate any fields that you have the tools for.
Do not stop calling tools until all the fields are validated.
Request for the proper input for the fields that are not validated.
After all the fields are validated, return in the provided json schema in proper format that can be parsed.
Don't add any other message.
Don't add ```json ``` tag either.
"""

def bank_agent_node(state: dict) -> dict:    
    # print("Account Opening Agent")
    # last_message = state["messages"][-1]
    print("Started Tool Calling")
        
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
# graph_builder.add_node("router", router)
graph_builder.add_node("bank", bank_agent_node)
graph_builder.add_node("tools",tool_node)
graph_builder.add_node("open_account", open_account)
graph_builder.add_edge(START, "chatbot")
# graph_builder.add_edge("chatbot", "router")
graph_builder.add_conditional_edges(
        "chatbot",
        lambda state: state.get("message_type"),
        { "bank": "bank", "general": END }
    )
graph_builder.add_conditional_edges("bank", tools_condition, {"tools": "tools","__end__": "open_account"})
graph_builder.add_edge("tools", "bank")

# graph_builder.add_edge("bank", END)

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