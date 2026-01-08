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

from indic_text_normalization.te.graph_utils import (
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_TE_DIGIT,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)
from indic_text_normalization.te.utils import get_abs_path

TE_ZERO_DIGIT = pynini.union("0", "౦")
TE_MOBILE_START_DIGITS = pynini.union("౬", "౭", "౮", "౯", "6", "7", "8", "9").optimize()
TE_LANDLINE_START_DIGITS = pynini.union("౨", "౩", "౪", "౬", "2", "3", "4", "6").optimize()

# Logic to handle optional leading zero
leading_zero = pynini.closure(pynini.cross(TE_ZERO_DIGIT, "సున్నా") + insert_space, 0, 1)

# Load the number mappings from the TSV file
digit_to_word = pynini.string_file(get_abs_path("data/telephone/number.tsv"))
digits = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
mobile_context = pynini.string_file(get_abs_path("data/telephone/mobile_context.tsv"))
landline_context = pynini.string_file(get_abs_path("data/telephone/landline_context.tsv"))
credit_context = pynini.string_file(get_abs_path("data/telephone/credit_context.tsv"))
pincode_context = pynini.string_file(get_abs_path("data/telephone/pincode_context.tsv"))

# Pattern to match any digit (Arabic or Telugu) for telephone numbers
any_digit = pynini.union(NEMO_DIGIT, NEMO_TE_DIGIT)

# Reusable optimized graph for any digit token
ascii_digit_to_word = pynini.string_map(
    [
        ("0", "సున్నా"),
        ("1", "ఒకటి"),
        ("2", "రెండు"),
        ("3", "మూడు"),
        ("4", "నాలుగు"),
        ("5", "ఐదు"),
        ("6", "ఆరు"),
        ("7", "ఏడు"),
        ("8", "ఎనిమిది"),
        ("9", "తొమ్మిది"),
    ]
).optimize()
num_token = pynini.union(digit_to_word, digits, zero, ascii_digit_to_word).optimize()


def generate_mobile(context_keywords: pynini.Fst) -> pynini.Fst:
    context_before, context_after = get_context(context_keywords)

    # Filter cardinals to only include allowed digits
    mobile_start_digit = pynini.union(
        TE_MOBILE_START_DIGITS @ digits, 
        TE_MOBILE_START_DIGITS @ digit_to_word,
        TE_MOBILE_START_DIGITS @ ascii_digit_to_word
    )

    country_code_digits = pynini.closure(num_token + insert_space, 1, 3)
    country_code = (
        pynutil.insert("country_code: \"")
        + context_before
        + pynini.cross("+", "ప్లస్")
        + insert_space
        + country_code_digits
        + pynutil.insert("\" ")
        + pynini.closure(delete_space, 0, 1)
    )

    extension_optional = pynini.closure(
        pynutil.insert("extension: \"")
        + pynini.closure(num_token + insert_space, 1, 3)
        + context_after
        + pynutil.insert("\" ")
        + delete_space,
        0,
        1,
    )

    number_part = mobile_start_digit + insert_space + pynini.closure(num_token + insert_space, 9)

    number_without_country = (
        pynutil.insert("number_part: \"")
        + context_before
        + leading_zero
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
    landline_start_digit = pynini.union(
        TE_LANDLINE_START_DIGITS @ digits, 
        TE_LANDLINE_START_DIGITS @ digit_to_word,
        TE_LANDLINE_START_DIGITS @ ascii_digit_to_word
    )

    std_code_graph = (
        leading_zero + pynini.closure(num_token + insert_space, std_length, std_length)
    )

    landline_digit_count = 9 - std_length
    landline_graph = (
        landline_start_digit
        + insert_space
        + pynini.closure(num_token + insert_space, landline_digit_count, landline_digit_count)
    )

    separator_optional = pynini.closure(pynini.union(pynini.cross("-", ""), pynini.cross(".", "")), 0, 1)

    std_code_in_brackets = (
        leading_zero
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

    all_digits = pynini.union(NEMO_TE_DIGIT, NEMO_DIGIT)

    non_digit_char = pynini.difference(NEMO_CHAR, pynini.union(all_digits, NEMO_WHITE_SPACE))
    word = pynini.closure(non_digit_char, 1) + pynini.accep(NEMO_SPACE)

    window = pynini.closure(word, 0, 5)

    before = pynini.closure(keywords + pynini.accep(NEMO_SPACE) + window, 0, 1)

    after = pynini.closure(pynutil.delete(NEMO_SPACE) + window + keywords, 0, 1)

    return before.optimize(), after.optimize()


def generate_credit(context_keywords: pynini.Fst) -> pynini.Fst:
    context_before, context_after = get_context(context_keywords)
    return (
        pynutil.insert("number_part: \"")
        + context_before
        + pynini.closure(num_token + insert_space, 4)
        + context_after
        + pynutil.insert("\" ")
        + delete_space
    ).optimize()


def generate_pincode(context_keywords: pynini.Fst) -> pynini.Fst:
    context_before, context_after = get_context(context_keywords)
    return (
        pynutil.insert("number_part: \"")
        + context_before
        + pynini.closure(num_token + insert_space, 6)
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
    digit_sequence_with_sep = (
        single_digit  # First digit (required)
        + pynini.closure(pynini.closure(separator, 0, 1) + single_digit, 6)  # At least 6 more digits
    )
    
    number_part_digits = consecutive_digits | digit_sequence_with_sep
    
    # Optional country code with + (with or without space after country code)
    country_code_digits = pynini.closure(single_digit, 1, 3)
    country_code_with_plus = (
        pynutil.insert("country_code: \"")
        + pynini.cross("+", "ప్లస్")
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
        + leading_zero
        + number_part_digits
        + pynutil.insert("\" ")
        + delete_space
    )
    
    return (pynini.union(number_with_country, number_without_country) + extension_optional).optimize()


class TelephoneFst(GraphFst):
    """
    Finite state transducer for tagging telephone numbers, e.g.
        ౯౧౫౭౧౧౪౦౦౭ -> telephone { number_part: "సున్నా తొమ్మిది ఒకటి ఐదు ఏడు ఒకటి ఒకటి నాలుగు సున్నా సున్నా ఏడు" }
        +౯౧ ౯౨౧౦౫౧౫౬౦౬ -> telephone { country_code: "ప్లస్ తొమ్మిది ఒకటి", number_part: "తొమ్మిది రెండు ఒకటి సున్నా ఐదు ఒకటి ఐదు ఆరు సున్నా ఆరు" }
        ౧౩౭౪-౩౦౯౯౮౮ -> telephone { number_part: "సున్నా ఒకటి మూడు ఏడు నాలుగు మూడు సున్నా తొమ్మిది తొమ్మిది ఎనిమిది ఎనిమిది" }

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
            pynutil.add_weight(mobile_number, 0.1)
            | pynutil.add_weight(landline, 0.1)
            | pynutil.add_weight(credit_card, 1.5)
            | pynutil.add_weight(pincode, 1.5)
            | pynutil.add_weight(general_telephone, 0.15)  # Fallback for edge cases
        )

        self.final = graph.optimize()
        self.fst = self.add_tokens(self.final)

