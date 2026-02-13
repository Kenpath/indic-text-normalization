import argparse
import sys
from pathlib import Path

# Add project root to sys.path to allow imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from indic_text_normalization.normalize import Normalizer

def load_test_cases(lang, category=None):
    """
    Load test cases from text file.
    Expected location: data/test_cases/<lang>.txt (relative to project root)
    """
    # Navigate to project root -> data -> test_cases
    exclude_file_name = "en_kaggle"
    if lang == exclude_file_name:
        return []
    
    file_path = project_root / "data" / "test_cases" / f"{lang}.txt"
    
    if not file_path.exists():
        print(f"Error: Test file not found: {file_path}")
        return []

    cases = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                parts = line.split("|||")
                if len(parts) >= 2:
                    inp = parts[0].strip()
                    cat = parts[1].strip()
                    
                    if category and cat != category:
                        continue
                        
                    cases.append({"input": inp, "category": cat, "line": line_num})
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []
        
    return cases

def run_tests():
    parser = argparse.ArgumentParser(description="Run text normalization tests")
    parser.add_argument("--lang", required=True, help="Language code (e.g., en, hi, ta)")
    parser.add_argument("--category", help="Run specific category (e.g., cardinal, time)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    
    args = parser.parse_args()
    
    print(f"Initializing Normalizer for language: {args.lang}")
    try:
        normalizer = Normalizer(lang=args.lang, input_case='cased')
    except Exception as e:
        print(f"Failed to initialize normalizer: {e}")
        return

    test_cases = load_test_cases(args.lang, args.category)
    if not test_cases:
        print(f"No test cases found for lang='{args.lang}'" + (f" category='{args.category}'" if args.category else ""))
        return

    print(f"Running {len(test_cases)} tests...")
    
    passed = 0
    failed = 0
    
    for case in test_cases:
        text = case["input"]
        cat = case["category"]
        
        try:
            result = normalizer.normalize(text)
            if args.verbose:
                print(f"Line {case['line']} [{cat}]: '{text}' -> '{result}'")
            passed += 1
        except Exception as e:
            failed += 1
            print(f"Line {case['line']} [{cat}] FAILED: '{text}'")
            print(f"  Error: {e}")

    print("-" * 30)
    print(f"Finished: {passed} passed, {failed} failed.")

if __name__ == "__main__":
    run_tests()
