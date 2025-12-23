# Test Cases for Text Normalization

This directory contains test cases for text normalization across all supported languages.

## File Structure

Each language has its own test file named `<lang>.txt`:

- `en.txt` - English
- `hi.txt` - Hindi
- `bn.txt` - Bengali
- `ta.txt` - Tamil
- `te.txt` - Telugu
- `gu.txt` - Gujarati
- `kn.txt` - Kannada
- `mr.txt` - Marathi
- `ma.txt` - Malayalam
- `ne.txt` - Nepali
- `sa.txt` - Sanskrit
- `bo.txt` - Bodo
- `do.txt` - Dogri
- `pu.txt` - Punjabi
- `as.txt` - Assamese
- `bho.txt` - Bhojpuri
- `mag.txt` - Magadhi
- `mai.txt` - Maithili
- `hne.txt` - Chhattisgarhi

## File Format

Each test file follows this format:

```
# Comments start with #
# Format: input|||category

# CARDINAL (examples)
123|||cardinal
१२३४|||cardinal

# TIME (examples)
12:30|||time
१२:३०|||time

# MONEY (examples)
₹100|||money
```

### Format Specification

- **Lines starting with `#`**: Comments (ignored by the test runner)
- **Test cases**: `input|||category`
  - `input`: The text to be normalized
  - `|||`: Delimiter (three pipe characters)
  - `category`: The category of the test (e.g., cardinal, time, money)

## Test Categories

Common categories across languages:

- `cardinal` - Cardinal numbers (e.g., 123, १२३४)
- `decimal` - Decimal numbers (e.g., 12.34, १२.३४)
- `fraction` - Fractions (e.g., 3/4, ३/४)
- `date` - Dates (e.g., 01-04-2024, ०१-०४-२०२४)
- `time` - Time expressions (e.g., 12:30, १२:३०)
- `money` - Currency amounts (e.g., ₹100, $50)
- `measure` - Measurements (e.g., 12 kg, १२ kg)
- `ordinal` - Ordinal numbers (e.g., 1st, १वाँ)
- `telephone` - Phone numbers
- `whitelist` - Abbreviations and special terms
- `mixed` - Mixed content with multiple categories
- `batch` - Batch testing examples

## Running Tests

To run tests for a specific language:

```bash
# From the project root
python tests/run_tests.py --lang hi

# Run specific category
python tests/run_tests.py --lang ta --category cardinal

# Verbose output
python tests/run_tests.py --lang en --verbose
```

See `tests/README.md` for more usage examples.

## Adding New Test Cases

1. Open the appropriate language file (e.g., `hi.txt` for Hindi)
2. Add new test cases following the format: `input|||category`
3. Group related tests under a comment header
4. Run tests to verify: `python tests/run_tests.py --lang <lang>`

### Example: Adding a new test to Hindi

```bash
# Edit data/test_cases/hi.txt
echo "५०० किलोमीटर|||measure" >> data/test_cases/hi.txt

# Test it
python tests/run_tests.py --lang hi --category measure
```

## Statistics

Total test cases: **2,215** across **19 languages**

| Language | Tests | File |
|----------|-------|------|
| English | 94 | en.txt |
| Hindi | 122 | hi.txt |
| Bengali | 97 | bn.txt |
| Tamil | 133 | ta.txt |
| Telugu | 116 | te.txt |
| Gujarati | 103 | gu.txt |
| Kannada | 117 | kn.txt |
| Marathi | 103 | mr.txt |
| Malayalam | 129 | ma.txt |
| Nepali | 125 | ne.txt |
| Sanskrit | 125 | sa.txt |
| Bodo | 125 | bo.txt |
| Dogri | 125 | do.txt |
| Punjabi | 125 | pu.txt |
| Assamese | 120 | as.txt |
| Bhojpuri | 103 | bho.txt |
| Magadhi | 103 | mag.txt |
| Maithili | 146 | mai.txt |
| Chhattisgarhi | 104 | hne.txt |

## Notes

- All test cases were extracted from the original test notebooks
- Test files use UTF-8 encoding to support all scripts
- Empty lines and comments are ignored by the test runner
- The delimiter `|||` was chosen to avoid conflicts with common text patterns