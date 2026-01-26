import re
from typing import Any

def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Parses the model output to extract the predicted answer letter for MMLU.
    
    The prompt instructs the model to respond with: "The correct answer is _".
    We look for this specific pattern using regex.
    """
    # Regex searches for "The correct answer is" followed optionally by whitespace
    # and then a single letter A-D. capture group 1 is the letter.
    match = re.search(r"The correct answer is\s*([A-D])", model_output)
    
    if match:
        return match.group(1)
    
    return None