from langchain_core.tools import tool
from utils import validate_nepali_phone_number
from langchain.tools import Tool
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