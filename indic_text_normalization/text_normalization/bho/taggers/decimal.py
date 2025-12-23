# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
    GraphFst,
    NEMO_DIGIT,
    insert_space,
)
from indic_text_normalization.text_normalization.bho.utils import get_abs_path

quantities = pynini.string_file(get_abs_path("data/numbers/thousands.tsv"))

# Convert Arabic digits (0-9) to Bhojpuri digits (०-९) - Devanagari script
arabic_to_bhojpuri_digit = pynini.string_map([
    ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
    ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९")
]).optimize()
arabic_to_bhojpuri_number = pynini.closure(arabic_to_bhojpuri_digit).optimize()


def get_quantity(decimal: 'pynini.FstLike', cardinal_up_to_hundred: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. १ लाख -> integer_part: "एक" quantity: "लाख"
    e.g. १.५ लाख -> integer_part: "एक" fractional_part: "पाँच" quantity: "लाख"

    Args:
        decimal: decimal FST
        cardinal_up_to_hundred: cardinal FST
    """
    numbers = cardinal_up_to_hundred

    res = (
        pynutil.insert("integer_part: \"")
        + numbers
        + pynutil.insert("\"")
        + insert_space
        + pynutil.insert("quantity: \"")
        + quantities
        + pynutil.insert("\"")
    )
    res |= decimal + insert_space + pynutil.insert("quantity: \"") + quantities + pynutil.insert("\"")
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g.
        -१२.५००६ करोड़ -> decimal { negative: "true" integer_part: "बारह"  fractional_part: "पाँच शून्य शून्य छह" quantity: "करोड़" }
        १ करोड़ -> decimal { integer_part: "एक" quantity: "करोड़" }

    cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        graph_digit = cardinal.digit | cardinal.zero
        cardinal_graph = cardinal.final_graph

        # Bhojpuri digit sequence: Bhojpuri digits → words with spaces
        bhojpuri_digit_sequence = (graph_digit + pynini.closure(insert_space + graph_digit)).optimize()
        # Arabic digit sequence: Arabic digits → convert to Bhojpuri → apply same sequence
        arabic_digit_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_digit_sequence = pynini.compose(
            arabic_digit_input,
            arabic_to_bhojpuri_number @ bhojpuri_digit_sequence,
        ).optimize()
        self.graph = (bhojpuri_digit_sequence | arabic_digit_sequence).optimize()

        # Handle both "." and "," as decimal separators (common in Indian number systems)
        point = pynutil.delete(pynini.union(".", ","))

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + insert_space,
            0,
            1,
        )

        self.graph_fractional = pynutil.insert("fractional_part: \"") + self.graph + pynutil.insert("\"")
        # Integer part uses cardinal_graph directly (already handles both Bhojpuri and Arabic digits)
        self.graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")

        # Pattern: integer_part + decimal_point + fractional_part
        # This handles both Bhojpuri digits (e.g., १२.३४) and Arabic digits (e.g., 12.34)
        final_graph_wo_sign = self.graph_integer + point + insert_space + self.graph_fractional

        self.final_graph_wo_negative = final_graph_wo_sign | get_quantity(final_graph_wo_sign, cardinal_graph)

        final_graph = optional_graph_negative + self.final_graph_wo_negative

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

