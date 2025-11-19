#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script for Tamil text normalization - All Categories
Run this script to test all Tamil FSTs
"""

import sys
import os
import io
import re

# Force UTF-8 encoding for Windows terminals
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add the nemo_text_processing to path
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to path so we can import nemo_text_processing
sys.path.insert(0, script_dir)

from nemo_text_processing.text_normalization.normalize import Normalizer


def arabic_to_tamil_digits(text):
    """Convert Arabic digits (0-9) to Tamil digits (௦-௯)"""
    arabic_to_tamil = {
        '0': '௦', '1': '௧', '2': '௨', '3': '௩', '4': '௪',
        '5': '௫', '6': '௬', '7': '௭', '8': '௮', '9': '௯'
    }
    result = ''
    for char in text:
        result += arabic_to_tamil.get(char, char)
    return result


def test_tamil_all():
    """Test Tamil text normalization with all categories - 5 mixed test cases"""
    
    try:
        normalizer = Normalizer(input_case="cased", lang="ta", deterministic=True)
    except Exception as e:
        print(f"Error initializing Normalizer: {str(e)}")
        return 1
    
    # 5 comprehensive test cases covering all categories (excluding whitelist)
    # First test with simple cases to verify normalization works
    test_cases = [
        "100",  # Simple number test
        "12:30",  # Simple time test
        "₹100",  # Simple money test
        "நான் 1 நாளில் 12:30 மணிக்கு ₹100 வாங்கினேன்",  # Numbers + Time + Money
        "இன்று 3வது நாளில் ₹2500.50 செலவழித்தேன்",  # Ordinal + Money + Decimal
        "காலை 9:15 மணிக்கு 25 புத்தகங்கள் ₹500க்கு வாங்கினேன்",  # Time + Numbers + Money
        "என் 2வது பிறந்தநாளில் ₹1000.75 பரிசு பெற்றேன்",  # Ordinal + Money + Decimal
        "மாலை 6:45 மணிக்கு 10.5 கிலோ அரிசி ₹350க்கு வாங்கினேன்"  # Time + Decimal + Money
    ]
    
    for input_text in test_cases:
        try:
            # Convert Arabic digits to Tamil digits
            tamil_input = arabic_to_tamil_digits(input_text)
            
            # Check for various patterns (validation that categories are present)
            has_numbers = bool(re.search(r'\d+', tamil_input))
            has_money = bool(re.search(r'[₹$€£¥]\d+', tamil_input))
            has_time = bool(re.search(r'\d+:\d+', tamil_input))
            has_whitelist = bool(re.search(r'(டாக்|புரோ|இஞ்|லெ|வை|கு|மா)\.', tamil_input))
            has_ordinal = bool(re.search(r'\d+(வது|ஆம்)', tamil_input))
            has_decimal = bool(re.search(r'\d+[.,]\d+', tamil_input))
            
            # Get normalized output - try with verbose mode first to see what's happening
            try:
                # Try normalizing the original input (with Arabic digits)
                normalized_output = normalizer.normalize(input_text, verbose=False, punct_post_process=True)
                if normalized_output is None:
                    normalized_output = "(None)"
                elif normalized_output == "":
                    normalized_output = "(empty string)"
                # If output is same as input, try with Tamil digits version
                if normalized_output == input_text:
                    normalized_output_tamil = normalizer.normalize(tamil_input, verbose=False, punct_post_process=True)
                    if normalized_output_tamil and normalized_output_tamil != tamil_input:
                        normalized_output = normalized_output_tamil
            except Exception as norm_error:
                normalized_output = f"(Error: {str(norm_error)})"
            
            # Print only Input, Output, and Categories
            print(f"Input:  {input_text}")
            print(f"Output: {normalized_output}")
            print(f"Categories: Numbers={has_numbers or has_ordinal or has_decimal}, "
                  f"Money={has_money}, Time={has_time}, Whitelist={has_whitelist}")
            print()
        except Exception as e:
            print(f"Input:  {input_text}")
            print(f"Output: (Error)")
            print(f"Categories: Error={str(e)}")
            print()
    
    return 0


if __name__ == "__main__":
    test_tamil_all()
