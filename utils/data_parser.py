import numpy as np
import re

def parse_data(data_str):
    """
    Parse string data into numerical values.
    
    Args:
        data_str (str): Input data string in various formats
        
    Returns:
        list: List of parsed float values
    """
    if not data_str or data_str.strip() == "":
        return []
    
    # Normalize decimal separators (replace commas with dots)
    data_str = data_str.replace(',', '.')
    
    # Check if each line contains a single number (one number per line)
    if '\n' in data_str:
        lines = data_str.strip().split('\n')
        # Try to parse each line as a single number first
        try:
            values = []
            for line in lines:
                line = line.strip()
                if line:  # Skip empty lines
                    values.append(float(line))
            return values
        except ValueError:
            # If that fails, continue with other parsing methods
            pass
    
    # Try to determine the separator for x,y pairs
    if '\n' in data_str and (',' in data_str or '\t' in data_str or ' ' in data_str):
        # Each line probably contains an x,y pair
        lines = data_str.strip().split('\n')
        values = []
        
        for line in lines:
            # Try to extract the first value from each line
            if ',' in line:
                parts = line.split(',')
                if parts and parts[0].strip():
                    values.append(float(parts[0].strip()))
            elif '\t' in line:
                parts = line.split('\t')
                if parts and parts[0].strip():
                    values.append(float(parts[0].strip()))
            else:
                parts = line.split()
                if parts and parts[0].strip():
                    values.append(float(parts[0].strip()))
        
        return values
    
    # Check if there are commas surrounded by digits/whitespace
    if re.search(r'\d\s*[,\.]\s*\d', data_str):
        # Comma/period separated values
        # Replace any sequence of whitespace around commas/periods with nothing
        data_str = re.sub(r'\s*[,\.]\s*', ',', data_str)
        values = [float(x) for x in data_str.split(',') if x.strip()]
        return values
    
    # Default: assume space-separated values
    values = [float(x) for x in data_str.split() if x.strip()]
    return values

def parse_error_data(error_str, expected_length):
    """
    Parse error data string into numerical values.
    
    Args:
        error_str (str): Input error data string
        expected_length (int): Expected number of values
        
    Returns:
        list: List of parsed float values, or None if input is empty
    """
    if not error_str or error_str.strip() == "":
        return None
    
    values = parse_data(error_str)
    
    # If we have a single value, replicate it to match expected length
    if len(values) == 1 and expected_length > 1:
        return [values[0]] * expected_length
    
    # Make sure we have the correct number of error values
    if len(values) != expected_length:
        raise ValueError(f"Number of error values ({len(values)}) does not match data points ({expected_length})")
    
    return values
