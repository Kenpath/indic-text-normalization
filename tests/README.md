# Text Normalization Tests

This directory contains a unified testing framework for text normalization across all supported languages.

## Structure

- **Test Data**: Stored in `data/test_cases/<lang>.txt` (in project root)
- **Test Script**: Single unified script `tests/run_tests.py`

## Usage

### Basic Usage
Run all tests for a specific language:
```bash
python tests/run_tests.py --lang hi
```

### Filter by Category
Run tests for a specific category only:
```bash
python tests/run_tests.py --lang ta --category cardinal
```

### Verbose Output
Show detailed output for each test:
```bash
python tests/run_tests.py --lang en --verbose
```

## Test Data Format
Test cases are stored in plain text files: `data/test_cases/<lang>.txt`

Format: `input|||category`

Example (`hi.txt`):
```text
123|||cardinal
12:30|||time
₹100|||money
```

## Adding New Tests
1. Open or create `data/test_cases/<lang>.txt`.
2. Add new lines in the format `input|||category`.
3. Run the validaton command.
