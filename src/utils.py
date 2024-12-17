# utils.py

import re
import logging
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def parse_filing(file_path):
    """
    Parse the 10-Q filing from the given file path by identifying the <DOCUMENT>
    with <TYPE>10-Q and extracting the <TEXT> content. Then extract relevant
    sections from that content.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return None

    soup = BeautifulSoup(content, 'lxml')

    # Find all <DOCUMENT> sections
    document_tags = soup.find_all('document')
    if not document_tags:
        logging.warning(f"No <DOCUMENT> tags found in {file_path}.")
        return None

    ten_q_document = None
    for doc in document_tags:
        type_tag = doc.find('type')
        if type_tag and '10-q' in type_tag.get_text(strip=True).lower():
            ten_q_document = doc
            break

    if ten_q_document is None:
        logging.warning(f"No 10-Q <DOCUMENT> section found in {file_path}.")
        return None

    text_tag = ten_q_document.find('text')
    if not text_tag:
        logging.warning(f"No <TEXT> tag found within the 10-Q document for {file_path}.")
        return None

    raw_text = text_tag.get_text(separator='\n')
    if not raw_text.strip():
        logging.warning(f"Extracted text is empty for {file_path}.")
        return None

    # Now we attempt to extract relevant sections (Risk Factors, MD&A)
    relevant_text = extract_relevant_sections(raw_text)
    if not relevant_text.strip():
        logging.warning(f"No relevant sections found in the extracted text for {file_path}.")
        return None

    return relevant_text

def extract_relevant_sections(text):
    """
    Extract key sections like "Risk Factors" and "Management’s Discussion and Analysis"
    from the given text. Tries multiple patterns and returns combined sections.
    """

    # Patterns to identify sections. We broaden the patterns to capture variations.
    # We look for "Item 1A. Risk Factors" and "Item 2. Management's Discussion and Analysis",
    # and also variations without item numbers.
    patterns = [
        r'(?i)(Item\s*1A\s*:\s*Risk\s+Factors.*?)(?=Item\s*\d|$)',
        r'(?i)(Item\s*1A\s*Risk\s+Factors.*?)(?=Item\s*\d|$)',
        r'(?i)(Risk\s+Factors.*?)(?=Item\s*\d|$)',
        r'(?i)(Item\s*2\s*:\s*Management\'?s\s*Discussion\s*and\s*Analysis.*?)(?=Item\s*\d|$)',
        r'(?i)(Item\s*2\s*Management\'?s\s*Discussion\s*and\s*Analysis.*?)(?=Item\s*\d|$)',
        r'(?i)(Management\'?s\s*Discussion\s*and\s*Analysis.*?)(?=Item\s*\d|$)'
    ]

    sections = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            # Strip to clean up whitespace
            clean_match = match.strip()
            if clean_match and clean_match not in sections:
                sections.append(clean_match)

    # If no sections found, consider returning the entire text as a fallback
    # This ensures we don't end up with nothing in case of unexpected formatting.
    if not sections:
        logging.info("No recognized sections found; using full text as fallback.")
        return text.strip()

    return ' '.join(sections)

def clean_text(text):
    # Remove HTML entities and excessive whitespace
    text = re.sub(r'&[\w#]+;', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
