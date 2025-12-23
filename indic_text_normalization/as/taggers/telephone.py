# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pynini
from pynini.lib import pynutil

from ..graph_utils import (
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_HI_DIGIT,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)
from ..utils import get_abs_path

HI_ZERO_DIGIT = pynini.union("0", "०")
HI_MOBILE_START_DIGITS = pynini.union("६", "७", "८", "९", "6", "7", "8", "9").optimize()
HI_LANDLINE_START_DIGITS = pynini.union("२", "३", "४", "६", "2", "3", "4", "6").optimize()

delete_zero = pynutil.delete(HI_ZERO_DIGIT)
delete_zero_optional = pynini.closure(delete_zero, 0, 1)
insert_shunya = pynutil.insert('शून्य') + insert_space

# Load the number mappings from the TSV file
digit_to_word = pynini.string_file(get_abs_path("data/telephone/number.tsv"))
digits = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
mobile_context = pynini.string_file(get_abs_path("data/telephone/mobile_context.tsv"))
landline_context = pynini.string_file(get_abs_path("data/telephone/landline_context.tsv"))
credit_context = pynini.string_file(get_abs_path("data/telephone/credit_context.tsv"))
pincode_context = pynini.string_file(get_abs_path("data/telephone/pincode_context.tsv"))

# Convert Arabic digits (0-9) to Hindi digits (०-९) for pattern matching
arabic_to_hindi_digit = pynini.string_map([
    ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
    ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९")
]).optimize()

# Reusable optimized graph for any digit token
# Supports both Arabic digits (via digit_to_word) and Hindi digits (via digits)
num_token = pynini.union(digit_to_word, digits, zero).optimize()

# Pattern to match any digit (Arabic or Hindi) for telephone numbers
any_digit = pynini.union(NEMO_DIGIT, NEMO_HI_DIGIT)


def generate_mobile(context_keywords: pynini.Fst) -> pynini.Fst:
    context_before, context_after = get_context(context_keywords)

    # Filter cardinals to only include allowed digits
    # Support both Arabic and Hindi digits for mobile start digits
    mobile_start_digit = pynini.union(
        HI_MOBILE_START_DIGITS @ digits, 
        HI_MOBILE_START_DIGITS @ digit_to_word
    )

    # Country code: match Arabic or Hindi digits, then convert to words
    # Accept 1-3 digits (Arabic or Hindi), convert to words digit-by-digit
    # Handle both with and without space after country code
    country_code_digits = pynini.closure(pynini.compose(any_digit, num_token) + insert_space, 1, 3)
    country_code = (
        pynutil.insert("country_code: \"")
        + context_before
        + pynini.cross("+", "प्लस")
        + insert_space
        + country_code_digits
        + pynutil.insert("\" ")
        + pynini.closure(delete_space, 0, 1)  # Optional space after country code
    )

    extension_optional = pynini.closure(
        pynutil.insert("extension: \"")
        + pynini.closure(pynini.compose(any_digit, num_token) + insert_space, 1, 3)
        + context_after
        + pynutil.insert("\" ")
        + delete_space,
        0,
        1,
    )

    # Number part: match Arabic or Hindi digits, then convert to words digit-by-digit
    # Mobile number: 1 start digit + 9 more digits = 10 digits total
    # First digit must be a mobile start digit (6, 7, 8, 9)
    mobile_first_digit = pynini.compose(HI_MOBILE_START_DIGITS, num_token) + insert_space
    remaining_digits = pynini.closure(pynini.compose(any_digit, num_token) + insert_space, 9)
    number_part = mobile_first_digit + remaining_digits

    number_without_country = (
        pynutil.insert("number_part: \"")
        + context_before
        + delete_zero_optional
        + insert_shunya
        + number_part
        + context_after
        + pynutil.insert("\" ")
        + delete_space
    )

    number_with_country = (
        country_code
        + pynutil.insert("number_part: \"")
        + number_part
        + context_after
        + pynutil.insert("\" ")
        + delete_space
    )

    return (pynini.union(number_with_country, number_without_country) + extension_optional).optimize()


def get_landline(std_length: int, context_keywords: pynini.Fst) -> pynini.Fst:
    context_before, context_after = get_context(context_keywords)

    # Filter cardinals to only include allowed digits
    # Support both Arabic and Hindi digits
    landline_start_digit = pynini.union(HI_LANDLINE_START_DIGITS @ digits, HI_LANDLINE_START_DIGITS @ digit_to_word)

    # STD code: match Arabic or Hindi digits, convert to words
    std_code_graph = (
        delete_zero_optional + insert_shunya + pynini.closure(pynini.compose(any_digit, num_token) + insert_space, std_length, std_length)
    )

    landline_digit_count = 9 - std_length
    # Landline number: match Arabic or Hindi digits, convert to words
    # First digit must be a landline start digit
    landline_first_digit = pynini.compose(HI_LANDLINE_START_DIGITS, num_token) + insert_space
    landline_remaining = pynini.closure(pynini.compose(any_digit, num_token) + insert_space, landline_digit_count - 1)
    landline_graph = landline_first_digit + landline_remaining

    separator_optional = pynini.closure(pynini.union(pynini.cross("-", ""), pynini.cross(".", "")), 0, 1)

    std_code_in_brackets = (
        delete_zero_optional
        + delete_space
        + pynutil.delete("(")
        + pynini.closure(delete_space, 0, 1)
        + std_code_graph
        + pynini.closure(delete_space, 0, 1)
        + pynutil.delete(")")
    )

    std_part = pynini.union(std_code_graph, std_code_in_brackets)

    return (
        pynutil.insert("number_part: \"")
        + context_before
        + std_part
        + separator_optional
        + delete_space
        + landline_graph
        + context_after
        + pynutil.insert("\" ")
    ).optimize()


def generate_landline(context_keywords: pynini.Fst) -> pynini.Fst:
    graph = (
        get_landline(2, context_keywords)
        | get_landline(3, context_keywords)
        | get_landline(4, context_keywords)
        | get_landline(5, context_keywords)
        | get_landline(6, context_keywords)
        | get_landline(7, context_keywords)
    )

    return graph.optimize()


def get_context(keywords: pynini.Fst):

    all_digits = pynini.union(NEMO_HI_DIGIT, NEMO_DIGIT)

    non_digit_char = pynini.difference(NEMO_CHAR, pynini.union(all_digits, NEMO_WHITE_SPACE))
    word = pynini.closure(non_digit_char, 1) + pynini.accep(NEMO_SPACE)

    window = pynini.closure(word, 0, 5)

    before = pynini.closure(keywords + pynini.accep(NEMO_SPACE) + window, 0, 1)

    after = pynini.closure(pynutil.delete(NEMO_SPACE) + window + keywords, 0, 1)

    return before.optimize(), after.optimize()


def generate_credit(context_keywords: pynini.Fst) -> pynini.Fst:
    context_before, context_after = get_context(context_keywords)
    # Credit card: match Arabic or Hindi digits, convert to words digit-by-digit
    credit_digits = pynini.closure(pynini.compose(any_digit, num_token) + insert_space, 4)
    return (
        pynutil.insert("number_part: \"")
        + context_before
        + credit_digits
        + context_after
        + pynutil.insert("\" ")
        + delete_space
    ).optimize()


def generate_pincode(context_keywords: pynini.Fst) -> pynini.Fst:
    context_before, context_after = get_context(context_keywords)
    # Pincode: match Arabic or Hindi digits, convert to words digit-by-digit
    pincode_digits = pynini.closure(pynini.compose(any_digit, num_token) + insert_space, 6)
    return (
        pynutil.insert("number_part: \"")
        + context_before
        + pincode_digits
        + context_after
        + pynutil.insert("\" ")
        + delete_space
    ).optimize()


def generate_general_telephone() -> pynini.Fst:
    """
    General telephone number pattern that matches any sequence of digits
    with +, -, spaces and converts them digit-by-digit.
    This handles edge cases that don't match specific mobile/landline patterns.
    Minimum 7 digits to avoid matching short numbers.
    """
    # Single digit conversion
    single_digit = pynini.compose(any_digit, num_token) + insert_space
    
    # Separators: - or . (deleted, not converted)
    separator = pynini.union(
        pynini.cross("-", ""),
        pynini.cross(".", ""),
    )
    
    # Number part: at least 7 digits (can have separators)
    # Pattern 1: 7+ consecutive digits (no separators)
    consecutive_digits = pynini.closure(single_digit, 7)
    
    # Pattern 2: digits with separators (at least 7 digits total)
    # Pattern: digit (separator? digit)* ensuring at least 7 digits
    # We build: first digit + (separator? + digit)* with at least 6 more digits
    digit_sequence_with_sep = (
        single_digit  # First digit (required)
        + pynini.closure(pynini.closure(separator, 0, 1) + single_digit, 6)  # At least 6 more digits (with optional separators)
    )
    
    number_part_digits = consecutive_digits | digit_sequence_with_sep
    
    # Optional country code with + (with or without space after country code)
    country_code_digits = pynini.closure(single_digit, 1, 3)
    country_code_with_plus = (
        pynutil.insert("country_code: \"")
        + pynini.cross("+", "प्लस")
        + insert_space
        + country_code_digits
        + pynutil.insert("\" ")
        + pynini.closure(delete_space, 0, 1)  # Optional space after country code
    )
    
    # Optional extension at the end (1-3 digits after space)
    extension_optional = pynini.closure(
        pynutil.insert("extension: \"")
        + pynini.closure(single_digit, 1, 3)
        + pynutil.insert("\" ")
        + delete_space,
        0,
        1,
    )
    
    # Number with country code (no leading zero handling - country code handles it)
    number_with_country = (
        country_code_with_plus
        + pynutil.insert("number_part: \"")
        + number_part_digits
        + pynutil.insert("\" ")
        + delete_space
    )
    
    # Number without country code (handle leading zero if present)
    number_without_country = (
        pynutil.insert("number_part: \"")
        + delete_zero_optional
        + insert_shunya
        + number_part_digits
        + pynutil.insert("\" ")
        + delete_space
    )
    
    return (pynini.union(number_with_country, number_without_country) + extension_optional).optimize()


class TelephoneFst(GraphFst):
    """
    Finite state transducer for tagging telephone numbers, e.g.
        ९१५७११४००७ -> telephone { number_part: "शून्य नौ एक पाँच सात एक एक चार शून्य शून्य सात" }
        +९१ ९२१०५१५६०६ -> telephone { country_code: "प्लस नौ एक", number_part: "नौ दो एक शून्य पाँच एक पाँच छह शून्य छह" }
        १३७४-३०९९८८ -> telephone { number_part: "शून्य एक तीन सात चार तीन शून्य नौ नौ आठ आठ" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization
    """

    def __init__(self):
        super().__init__(name="telephone", kind="classify")

        mobile_number = generate_mobile(mobile_context)
        landline = generate_landline(landline_context)
        credit_card = generate_credit(credit_context)
        pincode = generate_pincode(pincode_context)

        general_telephone = generate_general_telephone()
        
        graph = (
            pynutil.add_weight(mobile_number, 0.7)  # Specific patterns first
            | pynutil.add_weight(landline, 0.8)
            | pynutil.add_weight(credit_card, 0.9)
            | pynutil.add_weight(pincode, 1)
            | pynutil.add_weight(general_telephone, 0.75)  # Fallback for edge cases
        )

        self.final = graph.optimize()
        self.fst = self.add_tokens(self.final)
