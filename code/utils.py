import os
import json
from typing import List, Dict, Any, Tuple


def read_json(path: str) -> List[Dict[str, Any]]:
    """
    Read JSON data from a file, with each line containing a separate JSON object.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        List of parsed JSON objects
    """
    if not os.path.exists(path):
        return []
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading JSON from {path}: {e}")
        return []


def read_txt(path: str) -> str:
    """
    Read text content from a file.
    
    Args:
        path: Path to the text file
        
    Returns:
        String containing the file contents
    """
    if not os.path.exists(path):
        return ''
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = f.read()
        return data
    except Exception as e:
        print(f"Error reading text from {path}: {e}")
        return ''


def resume(path: str) -> Tuple[int, int, int, int, int, int]:
    """
    Resume processing from a checkpoint by analyzing existing results.
    
    Args:
        path: Path to the results file
        
    Returns:
        Tuple containing:
        - start_idx: Index to resume from
        - start_corr: Count of correct predictions
        - start_inco: Count of incorrect predictions
        - cause_idx: Count of causation queries processed
        - cause_corr: Count of correct causation predictions
        - cause_inco: Count of incorrect causation predictions
    """
    datas = read_json(path)
    start_idx = len(datas)
    start_corr = 0
    start_inco = 0
    cause_idx = 0
    cause_corr = 0
    cause_inco = 0
    
    for data in datas:
        is_causation = 'intentionally' not in data['query']
        
        if is_causation:
            cause_idx += 1
            
        if data['correct']:
            start_corr += 1
            if is_causation:
                cause_corr += 1
        else:
            start_inco += 1
            if is_causation:
                cause_inco += 1
                
    return start_idx, start_corr, start_inco, cause_idx, cause_corr, cause_inco


def partial_upper(text: str) -> str:
    """
    Replace boolean literals in JSON strings to Python format.
    
    Args:
        text: JSON string with potentially lowercase boolean values
        
    Returns:
        String with boolean values converted to Python format
    """
    # Convert JSON boolean literals to Python format
    return text.replace(': true', ': True').replace(': false', ': False')


def parse_json_response(response: str) -> Dict:
    """
    Parse JSON from model response, handling different formats.
    
    Args:
        response: Raw response string that may contain JSON
        
    Returns:
        Parsed JSON as dictionary
    """
    if response.startswith('```json'):
        # Extract JSON from code blocks
        clean_response = response.replace('```json', '').replace('```', '').strip()
    elif response.startswith('```'):
        # Handle code blocks without language specification
        clean_response = response.replace('```', '').strip()
    else:
        clean_response = response
    
    try:
        # Use partial_upper to handle potential case issues
        return eval(partial_upper(clean_response))
    except Exception as e:
        raise ValueError(f"Failed to parse JSON response: {e}")