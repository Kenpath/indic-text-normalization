# Indic Text Normalization

A production-ready, comprehensive Python package for text normalization across multiple Indian languages, built on Weighted Finite-State Transducers (WFST) using Pynini.

## Overview

`indic-text-normalization` is a production-ready, WFST-based library that provides **deterministic, low-latency, and explainable** text normalization for **TTS, ASR, and NLP** pipelines, converting semiotic entities (numbers, dates, currency) into spoken form across 19 Indian languages.

## Supported Languages

- **Hindi** (hi) - हिन्दी
- **Bengali** (bn) - বাংলা
- **Marathi** (mr) - मराठी
- **Telugu** (te) - తెలుగు
- **Kannada** (kn) - ಕನ್ನಡ
- **Bhojpuri** (bho) - भोजपुरी
- **Magahi** (mag) - मगही
- **Chhattisgarhi** (hne) - छत्तीसगढ़ी
- **Maithili** (mai) - मैथिली
- **Assamese** (ase) - অসমীয়া
- **Bodo** (bo) - बड़ो
- **Dogri** (do) - डोगरी
- **Gujarati** (gu) - ગુજરાતી
- **Malayalam** (ma) - മലയാളം
- **Punjabi** (pu) - ਪੰਜਾਬੀ
- **Tamil** (ta) - தமிழ்
- **English (Indian)** (en) - English
- **Nepali** (ne) - नेपाली
- **Sanskrit** (sa) - संस्कृत

## Features

### Semiotic Classes

Each language module supports normalization of:

- **Cardinal Numbers** - Both native script and Arabic numerals
- **Ordinal Numbers** - First, second, third, etc.
- **Decimal Numbers** - Decimal point verbalization
- **Fractions** - Proper fraction handling
- **Date** - Multiple date formats (DD/MM/YYYY, YYYY-MM-DD, etc.)
- **Time** - 12-hour and 24-hour formats
- **Telephone Numbers** - Digit-by-digit verbalization
- **Measurements** - Units with proper verbalization
- **Money** - Currency symbols and amounts
- **Electronic** - Email addresses, URLs, hashtags
- **Roman Numerals** - Conversion to spoken form
- **Abbreviations** - Common abbreviations expansion
- **Whitelist** - Custom word mappings

## Installation

For detailed installation instructions including platform-specific setup, troubleshooting, and multiple installation methods, please refer to the **[Setup Guide](setup.md)**.

**Quick Start:**

### Using Conda
```bash
conda create -n indic_tn python=3.10
conda activate indic_tn
conda install -c conda-forge pynini=2.1.6.post1
pip install -e .
```

### MacOS
```bash
brew install openfst
CPLUS_INCLUDE_PATH=/opt/homebrew/include LIBRARY_PATH=/opt/homebrew/lib pip install pynini
pip install -e .
```


## Usage

### Basic Usage

```python
from indic_text_normalization import Normalizer

# Initialize normalizer for a specific language
normalizer = Normalizer(input_case='cased', lang='hi')

# Normalize text
text = "मैं 25 साल का हूं और मेरा फोन नंबर 9876543210 है।"
normalized = normalizer.normalize(text)
print(normalized)
# Output: मैं पच्चीस साल का हूं और मेरा फोन नंबर नौ आठ सात छह पांच चार तीन दो एक शून्य है।
```

### Language-Specific Examples

**Tamil:**
```python
normalizer = Normalizer(input_case='cased', lang='ta')
text = "எனக்கு 1,234 ரூபாய் வேண்டும்"
normalized = normalizer.normalize(text)
```

**Malayalam:**
```python
normalizer = Normalizer(input_case='cased', lang='ma')
text = "സമയം 10:30 ആണ്"
normalized = normalizer.normalize(text)
```

**Gujarati:**
```python
normalizer = Normalizer(input_case='cased', lang='gu')
text = "તારીખ 15/08/2024 છે"
normalized = normalizer.normalize(text)
```

## Project Structure

```
indic-text-normalization/
├── indic_text_normalization/
│   ├── hi/          # Hindi
│   ├── ta/          # Tamil
│   ├── ma/          # Malayalam
│   ├── gu/          # Gujarati
│   ├── ase/         # Assamese
│   ├── mai/         # Maithili
│   ├── mag/         # Magadhi
│   ├── hne/         # Chhattisgarhi
│   ├── bo/          # Bodo
│   ├── do/          # Dogri
│   ├── pu/          # Punjabi
│   └── sa/          # Sanskrit
├── tests/           # Test notebooks and scripts
└── README.md
```

Each language directory contains:
- `taggers/` - FST-based taggers for each semiotic class
- `verbalizers/` - FST-based verbalizers for converting tagged text to spoken form
- `data/` - TSV files with language-specific terminology
- `tokenize_and_classify.py` - Main classification logic

## Testing

Test notebooks are available for each language:

```bash
# Run Jupyter notebook
jupyter notebook tests/test_hindi.ipynb
jupyter notebook tests/test_tamil.ipynb
# ... etc.
```

## Development

### Adding a New Language

1. Create a new directory under `indic_text_normalization/` with the language code
2. Replicate the structure from an existing language (e.g., Hindi)
3. Update all TSV data files with native terminology
4. Modify taggers and verbalizers for language-specific rules
5. Update `normalizer.py` to include the new language
6. Create test cases

### Contributing

Contributions are welcome! Please ensure:
- All data uses authentic native terminology
- Code follows existing patterns and structure
- Test cases are provided for new features
- Documentation is updated

## Technical Details

### WFST Architecture

The normalization pipeline uses Weighted Finite-State Transducers (WFST) implemented with Pynini:

1. **Tokenization** - Input text is tokenized
2. **Classification** - Each token is classified into semiotic classes
3. **Verbalization** - Classified tokens are converted to spoken form
4. **Post-processing** - Final cleanup and formatting

### Digit Handling

- **Arabic Digits** (0-9) - Converted to native script or verbalized directly
- **Native Script Digits** - Devanagari (०-९), Tamil (௦-௯), etc.
- **Mixed Input** - Handles both digit systems in the same text

## License

This project is licensed under the Apache 2.0 License.

## Acknowledgments

Built upon the foundation of NVIDIA NeMo Text Processing, adapted and extended for comprehensive Indic language support with authentic native terminology and language-specific normalization rules.

## Support

For questions, bug reports, or feature requests, please open an issue on the GitHub repository.
