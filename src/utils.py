# utils.py

import re
from bs4 import BeautifulSoup
import pandas as pd

def parse_filing(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Initialize BeautifulSoup for parsing
    soup = BeautifulSoup(content, 'lxml')
    
    # Remove script, style, and other non-relevant tags
    for tag in soup(['script', 'style', 'header', 'footer', 'img', 'meta', 'link']):
        tag.decompose()

    # Extract text and convert to DataFrame to filter rows
    text = soup.get_text(separator='\n')
    text_lines = text.splitlines()
    
    # Load into DataFrame for filtering
    df = pd.DataFrame(text_lines, columns=['line'])
    df['line'] = df['line'].str.strip()  # Remove extra whitespace
    df = df[df['line'] != '']            # Drop empty lines

    # Concatenate cleaned text
    cleaned_text = '\n'.join(df['line'])
    
    # Extract relevant sections using regex
    relevant_text = extract_relevant_sections(cleaned_text)
    return relevant_text

def extract_relevant_sections(text):
    # Define patterns to extract relevant sections
    patterns = [
        r'(?i)(Item\s*2\s*[\.:]?\s*Management\'?s\s+Discussion\s+and\s+Analysis.*?)(?=Item\s*\d|\Z)', 
        r'(?i)(Item\s*1A\s*[\.:]?\s*Risk\s+Factors.*?)(?=Item\s*\d|\Z)',
        r'(?i)(Quantitative\s+and\s+Qualitative\s+Disclosures\s+about\s+Market\s+Risk.*?)(?=Item\s*\d|\Z)',
        r'(?i)(Management\'?s\s+Discussion\s+and\s+Analysis.*?)(?=Item\s*\d|\Z)',  # Alternate phrasing without "Item 2"
        r'(?i)(Risk\s+Factors.*?)(?=Item\s*\d|\Z)',  # Alternate phrasing without "Item 1A"
    ]

    
    sections = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            sections.append(match)
    
    # Combine sections into one string
    return ' '.join(sections)

def clean_text(text):
    # Remove HTML entities and excessive whitespace
    text = re.sub(r'&[\w#]+;', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
