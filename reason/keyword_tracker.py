"""
Keyword tracking utilities for GSM8K problems.
Extracts names, quantities, and objects from problems and tracks their retention in KV cache.
"""

import re
from typing import List, Dict, Set, Tuple
from collections import defaultdict


def extract_keywords(question_text: str) -> Dict[str, List[str]]:
    """
    Extract keywords (names, quantities, objects) from GSM8K problem text.
    
    Parameters
    ----------
    question_text : str
        The GSM8K problem text
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary with keys: 'names', 'quantities', 'objects'
    """
    keywords = {
        'names': [],
        'quantities': [],
        'objects': []
    }
    
    # Extract quantities (numbers with units or standalone numbers)
    # Pattern: numbers, numbers with units (dollars, apples, etc.)
    quantity_patterns = [
        r'\$?\d+(?:\.\d+)?\s*(?:dollars?|cents?|apples?|oranges?|books?|pens?|pencils?|toys?|cars?|bikes?|cookies?|candies?|marbles?|stamps?|coins?|cards?|balls?|pieces?|items?|units?|pounds?|ounces?|kilograms?|grams?|meters?|kilometers?|miles?|hours?|minutes?|days?|weeks?|months?|years?)',
        r'\d+(?:\.\d+)?\s*(?:dollars?|cents?|apples?|oranges?|books?|pens?|pencils?|toys?|cars?|bikes?|cookies?|candies?|marbles?|stamps?|coins?|cards?|balls?|pieces?|items?|units?|pounds?|ounces?|kilograms?|grams?|meters?|kilometers?|miles?|hours?|minutes?|days?|weeks?|months?|years?)',
        r'\d+(?:\.\d+)?',  # Standalone numbers
    ]
    
    for pattern in quantity_patterns:
        matches = re.findall(pattern, question_text, re.IGNORECASE)
        keywords['quantities'].extend([m.strip() for m in matches if m.strip()])
    
    # Extract names (capitalized words that appear to be names)
    # Common name patterns in GSM8K
    name_patterns = [
        r'\b[A-Z][a-z]+\b',  # Capitalized words
    ]
    
    # Common names in GSM8K problems
    common_names = [
        'Alice', 'Bob', 'Charlie', 'David', 'Emma', 'Frank', 'Grace', 'Henry',
        'Ivy', 'Jack', 'Kate', 'Liam', 'Mary', 'Nancy', 'Oliver', 'Paul',
        'Quinn', 'Rachel', 'Sam', 'Tom', 'Uma', 'Victor', 'Wendy', 'Xavier',
        'Yara', 'Zoe', 'John', 'Jane', 'Mike', 'Sarah', 'Chris', 'Lisa',
        'Mark', 'Amy', 'Dan', 'Emily', 'James', 'Anna', 'Robert', 'Linda'
    ]
    
    for pattern in name_patterns:
        matches = re.findall(pattern, question_text)
        for match in matches:
            # Filter out common words that start with capital (like "The", "If", etc.)
            if match.lower() not in ['the', 'if', 'and', 'or', 'but', 'a', 'an', 'to', 'for', 'of', 'in', 'on', 'at', 'by', 'with']:
                if match in common_names or len(match) > 3:
                    keywords['names'].append(match)
    
    # Extract objects (common nouns that represent items)
    object_keywords = [
        'apple', 'apples', 'orange', 'oranges', 'book', 'books', 'pen', 'pens',
        'pencil', 'pencils', 'toy', 'toys', 'car', 'cars', 'bike', 'bikes',
        'cookie', 'cookies', 'candy', 'candies', 'marble', 'marbles', 'stamp', 'stamps',
        'coin', 'coins', 'card', 'cards', 'ball', 'balls', 'piece', 'pieces',
        'item', 'items', 'unit', 'units', 'box', 'boxes', 'bag', 'bags',
        'bottle', 'bottles', 'cup', 'cups', 'plate', 'plates', 'chair', 'chairs',
        'table', 'tables', 'desk', 'desks', 'room', 'rooms', 'house', 'houses',
        'store', 'stores', 'shop', 'shops', 'school', 'schools', 'class', 'classes',
        'student', 'students', 'teacher', 'teachers', 'friend', 'friends',
        'person', 'people', 'child', 'children', 'kid', 'kids'
    ]
    
    question_lower = question_text.lower()
    for obj in object_keywords:
        if obj in question_lower:
            # Find the actual word in context (with proper capitalization)
            pattern = r'\b' + re.escape(obj) + r'\b'
            matches = re.findall(pattern, question_text, re.IGNORECASE)
            keywords['objects'].extend([m for m in matches if m not in keywords['objects']])
    
    # Remove duplicates while preserving order
    for key in keywords:
        seen = set()
        keywords[key] = [x for x in keywords[key] if not (x in seen or seen.add(x))]
    
    return keywords


def tokenize_keywords(keywords: Dict[str, List[str]], tokenizer) -> Dict[str, Set[int]]:
    """
    Convert keyword strings to token IDs.
    
    Parameters
    ----------
    keywords : Dict[str, List[str]]
        Dictionary of keyword strings
    tokenizer : AutoTokenizer
        Tokenizer to use for encoding
        
    Returns
    -------
    Dict[str, Set[int]]
        Dictionary mapping keyword types to sets of token IDs
    """
    keyword_token_ids = {
        'names': set(),
        'quantities': set(),
        'objects': set()
    }
    
    for key_type, keyword_list in keywords.items():
        for keyword in keyword_list:
            # Tokenize the keyword
            tokens = tokenizer.encode(keyword, add_special_tokens=False)
            keyword_token_ids[key_type].update(tokens)
    
    return keyword_token_ids


def track_token_retention(
    all_token_ids: List[int],
    retained_token_indices: List[int],
    keyword_token_ids: Dict[str, Set[int]]
) -> Dict[str, Dict[str, any]]:
    """
    Track which keywords are retained or evicted from the KV cache.
    
    Parameters
    ----------
    all_token_ids : List[int]
        All token IDs in the sequence
    retained_token_indices : List[int]
        Indices of tokens that were retained in the cache
    keyword_token_ids : Dict[str, Set[int]]
        Token IDs for each keyword type
        
    Returns
    -------
    Dict[str, Dict[str, any]]
        Tracking results for each keyword type
    """
    retained_token_ids = set([all_token_ids[i] for i in retained_token_indices if i < len(all_token_ids)])
    evicted_token_ids = set(all_token_ids) - retained_token_ids
    
    results = {}
    
    for key_type, token_set in keyword_token_ids.items():
        retained_keywords = token_set & retained_token_ids
        evicted_keywords = token_set & evicted_token_ids
        
        results[key_type] = {
            'total_keyword_tokens': len(token_set),
            'retained_keyword_tokens': len(retained_keywords),
            'evicted_keyword_tokens': len(evicted_keywords),
            'retention_rate': len(retained_keywords) / len(token_set) if len(token_set) > 0 else 0.0,
            'retained_token_ids': list(retained_keywords),
            'evicted_token_ids': list(evicted_keywords)
        }
    
    return results

