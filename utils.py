import re

def validate_nepali_phone_number(phone_number: str) -> bool:
    # Remove spaces or other unwanted characters from the phone number
    phone_number = phone_number.strip()
    
    # Regex for validating Nepali phone number
    nepali_phone_regex = r"^\+9779[0-9]{1}[0-9]{8}$|^9[0-9]{1}[0-9]{8}$"

    # Match the phone number against the regex pattern
    if re.match(nepali_phone_regex, phone_number):
        return True
    return False