#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified test script for text normalization across all languages.

Usage:
    python tests/run_tests.py --lang hi
    python tests/run_tests.py --lang en --category cardinal --verbose
    python tests/run_tests.py --lang ta --validate
"""

import sys
import os
import argparse
import io
from pathlib import Path
from collections import defaultdict

# Force UTF-8 encoding for Windows terminals
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add the project root to path
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))


def load_test_cases(lang_code, category_filter=None):
    """Load test cases from the language-specific test file."""
    test_file = script_dir / 'data' / 'test_cases' / f'{lang_code}.txt'
    
    if not test_file.exists():
        print(f"Error: Test file not found for language '{lang_code}'")
        print(f"Expected location: {test_file}")
        return {}
    
    test_cases = defaultdict(list)
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse line: input|||category
            parts = line.split('|||')
            if len(parts) != 2:
                continue
            
            test_input, category = parts
            
            # Apply category filter if specified
            if category_filter and category != category_filter:
                continue
            
            test_cases[category].append(test_input)
    
    return dict(test_cases)


def run_tests(lang_code, test_cases, verbose=False, validate=False):
    """Run normalization tests for the given language."""
    
    # Import here to allow --help to work without dependencies
    try:
        from indic_text_normalization.text_normalization.normalize import Normalizer
    except ImportError as e:
        print(f"Error: Required dependencies not installed.")
        print(f"Please install requirements: pip install -r requirements.txt")
        print(f"Details: {e}")
        return 1
    
    print("="*60)
    print(f"Initializing {lang_code.upper()} Text Normalizer...")
    print("="*60)
    
    try:
        normalizer = Normalizer(
            input_case='cased',
            lang=lang_code,
            cache_dir=None,
            overwrite_cache=False,
            post_process=True
        )
        print(f"✓ Normalizer initialized successfully for language: {lang_code}")
    except Exception as e:
        print(f"✗ Error initializing normalizer: {e}")
        return 1
    
    # Statistics
    total_tests = sum(len(tests) for tests in test_cases.values())
    passed = 0
    errors = 0
    
    print(f"\nRunning {total_tests} tests across {len(test_cases)} categories...\n")
    
    # Run tests by category
    for category, inputs in sorted(test_cases.items()):
        print("="*60)
        print(f"{category.upper()} ({len(inputs)} tests)")
        print("="*60)
        
        for i, test_input in enumerate(inputs, 1):
            try:
                # Normalize the input
                output = normalizer.normalize(test_input, verbose=False, punct_post_process=True)
                
                if output is None:
                    output = test_input
                
                # Display results
                if verbose:
                    print(f"\n[{i}/{len(inputs)}]")
                    print(f"  Input:  {test_input}")
                    print(f"  Output: {output}")
                else:
                    # Compact format
                    print(f"  {test_input:30} -> {output}")
                
                # For validation mode, we would check against expected outputs
                # Since we don't have expected outputs stored, we just mark as passed
                # if normalization completed without error
                passed += 1
                
            except Exception as e:
                print(f"  ✗ Error normalizing '{test_input}': {e}")
                errors += 1
        
        print()  # Blank line between categories
    
    # Print summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total tests:  {total_tests}")
    print(f"Completed:    {passed}")
    print(f"Errors:       {errors}")
    
    if errors > 0:
        print(f"\n⚠ {errors} test(s) encountered errors")
        return 1
    else:
        print(f"\n✓ All tests completed successfully!")
        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run text normalization tests for any language',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests for Hindi
  python tests/run_tests.py --lang hi
  
  # Run specific category for Tamil with verbose output
  python tests/run_tests.py --lang ta --category cardinal --verbose
  
  # Run tests for English
  python tests/run_tests.py --lang en
  
Supported languages:
  en, hi, bn, ta, te, gu, kn, mr, ma, ne, sa, bo, do, pu, as, bho, mag, mai, hne
        """
    )
    
    parser.add_argument(
        '--lang',
        required=True,
        help='Language code (e.g., hi, en, ta, bn)'
    )
    
    parser.add_argument(
        '--category',
        help='Filter by specific category (e.g., cardinal, time, money)',
        default=None
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Show detailed output for each test'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        default=False,
        help='Validate outputs against expected results (future feature)'
    )
    
    args = parser.parse_args()
    
    # Load test cases
    test_cases = load_test_cases(args.lang, args.category)
    
    if not test_cases:
        print(f"No test cases found for language '{args.lang}'")
        if args.category:
            print(f"with category filter '{args.category}'")
        return 1
    
    # Run tests
    return run_tests(args.lang, test_cases, args.verbose, args.validate)


if __name__ == '__main__':
    sys.exit(main())

