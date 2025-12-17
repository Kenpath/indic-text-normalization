# Text Normalization Tests

This directory contains a unified testing framework for text normalization across all supported languages.

## Structure

The test system has been refactored to eliminate repetition:

- **Test Data**: Stored in `data/test_cases/<lang>.txt`
- **Test Script**: Single unified script `run_tests.py` that works for all languages

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

### Combined Options

```bash
python tests/run_tests.py --lang hi --category time --verbose
```

## Supported Languages

- `en` - English
- `hi` - Hindi
- `bn` - Bengali
- `ta` - Tamil
- `te` - Telugu
- `gu` - Gujarati
- `kn` - Kannada
- `mr` - Marathi
- `ma` - Malayalam
- `ne` - Nepali
- `sa` - Sanskrit
- `bo` - Bodo
- `do` - Dogri
- `Pu` - Punjabi
- `ase` - Assamese
- `bho` - Bhojpuri
- `mag` - Magadhi
- `mai` - Maithili
- `cg` - Chhattisgarhi

## Test Categories

Common categories across languages:
- `cardinal` - Cardinal numbers
- `decimal` - Decimal numbers
- `fraction` - Fractions
- `date` - Dates
- `time` - Time expressions
- `money` - Currency amounts
- `measure` - Measurements
- `ordinal` - Ordinal numbers
- `telephone` - Phone numbers
- `whitelist` - Abbreviations and special terms
- `mixed` - Mixed content with multiple categories
- `batch` - Batch testing examples

## Test Data Format

Test cases are stored in plain text format at:
```
data/test_cases/<lang>.txt
```

Format: `input|||category`

Example:
```
123|||cardinal
12:30|||time
₹100|||money
```

## Adding New Tests

To add new test cases for a language:

1. Open the test file: `data/test_cases/<lang>.txt`
2. Add new lines in the format: `input|||category`
3. Run the tests to verify: `python tests/run_tests.py --lang <lang>`

## Examples

### Test Hindi Cardinal Numbers
```bash
python tests/run_tests.py --lang hi --category cardinal
```

### Test Tamil Time Expressions with Verbose Output
```bash
python tests/run_tests.py --lang ta --category time --verbose
```

### Test All English Normalizations
```bash
python tests/run_tests.py --lang en
```

## Migration from Old Notebooks

The test system has been migrated from individual Jupyter notebooks to this unified approach:
- ✅ All test cases extracted and preserved
- ✅ Single reusable test script
- ✅ Easy to add/modify test cases
- ✅ Consistent testing across all languages
- ✅ Command-line interface for automation

