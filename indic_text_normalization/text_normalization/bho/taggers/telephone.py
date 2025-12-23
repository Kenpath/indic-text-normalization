# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from indic_text_normalization.text_normalization.bho.graph_utils import (
    NEMO_DIGIT,
    NEMO_BHO_DIGIT,
    GraphFst,
    delete_space,
    insert_space,
)
from indic_text_normalization.text_normalization.bho.utils import get_abs_path


# Load the number mappings from the TSV file (contains both Arabic and Bhojpuri digit mappings)
digit_to_word = pynini.string_file(get_abs_path("data/telephone/number.tsv"))

# Pattern to match any digit (Arabic or Bhojpuri/Devanagari)
any_digit = pynini.union(NEMO_DIGIT, NEMO_BHO_DIGIT)

# Single digit to word conversion
single_digit_to_word = (any_digit @ digit_to_word).optimize()


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers in Bhojpuri.

    Supports:
        - Phone numbers with separators (dash, dot)
        - International format with country code (+91, etc.)
        - Both Arabic (0-9) and Bhojpuri (०-९) digits
        - Minimum 7 digits to be recognized as telephone

    Examples:
        1374-309988 -> telephone { number_part: "एक तीन सात चार तीन शून्य नौ नौ आठ आठ" }
        १३७४-३०९९८८ -> telephone { number_part: "एक तीन सात चार तीन शून्य नौ नौ आठ आठ" }
        +91 9876543210 -> telephone { country_code: "प्लस नौ एक" number_part: "नौ आठ सात छह पाँच चार तीन दो एक शून्य" }

    Args:
        deterministic: if True will provide a single transduction option
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)

        # Single digit conversion (digit -> word + space)
        single_digit = single_digit_to_word + insert_space

        # Separators: - or . (deleted, not converted)
        separator = pynini.union(
            pynini.cross("-", ""),
            pynini.cross(".", ""),
        )

        # Pattern: digits with mandatory separator(s)
        # This ensures we only match phone-number-like patterns (with dashes/dots)
        # Format: digit+ separator digit+ (optionally more separator digit+)
        digit_group = pynini.closure(single_digit, 1)
        
        # Phone number with at least one separator: digit+ - digit+ (- digit+)*
        phone_with_separator = (
            digit_group
            + pynini.closure(separator + digit_group, 1)  # At least one separator required
        )

        # Phone number without separator but with 10+ consecutive digits (mobile/landline)
        consecutive_digits_10_plus = pynini.closure(single_digit, 10)

        # Country code with + 
        country_code_digits = pynini.closure(single_digit, 1, 3)
        country_code_with_plus = (
            pynutil.insert("country_code: \"")
            + pynini.cross("+", "प्लस")  # "plus" in Bhojpuri/Hindi
            + insert_space
            + country_code_digits
            + pynutil.insert("\" ")
            + pynini.closure(delete_space, 0, 1)
        )

        # Number part with separator (main pattern for phone numbers like 1374-309988)
        number_with_separator = (
            pynutil.insert("number_part: \"")
            + phone_with_separator
            + pynutil.insert("\"")
        )

        # Number part without separator (10+ digits like mobile numbers)
        number_consecutive = (
            pynutil.insert("number_part: \"")
            + consecutive_digits_10_plus
            + pynutil.insert("\"")
        )

        # Combine: with country code or without
        graph_with_country_sep = (
            country_code_with_plus
            + pynutil.insert("number_part: \"")
            + phone_with_separator
            + pynutil.insert("\"")
        )

        graph_with_country_consecutive = (
            country_code_with_plus
            + pynutil.insert("number_part: \"")
            + consecutive_digits_10_plus
            + pynutil.insert("\"")
        )

        graph = (
            pynutil.add_weight(graph_with_country_sep, 0.8)
            | pynutil.add_weight(graph_with_country_consecutive, 0.85)
            | pynutil.add_weight(number_with_separator, 0.9)
            | pynutil.add_weight(number_consecutive, 0.95)
        )

        self.final = graph.optimize()
        self.fst = self.add_tokens(self.final)
