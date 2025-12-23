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

from indic_text_normalization.text_normalization.bn.graph_utils import (
    NEMO_DIGIT,
    GraphFst,
    insert_space,
)
from indic_text_normalization.text_normalization.bn.utils import get_abs_path

quantities = pynini.string_file(get_abs_path("data/numbers/thousands.tsv"))

# Convert Arabic digits (0-9) to Bengali digits (০-৯)
arabic_to_bengali_digit = pynini.string_map([
    ("0", "০"), ("1", "১"), ("2", "২"), ("3", "৩"), ("4", "৪"),
    ("5", "৫"), ("6", "৬"), ("7", "৭"), ("8", "৮"), ("9", "৯")
]).optimize()
arabic_to_bengali_number = pynini.closure(arabic_to_bengali_digit).optimize()


def get_quantity(decimal: 'pynini.FstLike', cardinal_up_to_hundred: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. ১ লক্ষ -> integer_part: "এক" quantity: "লক্ষ"
    e.g. 1 লক্ষ -> integer_part: "এক" quantity: "লক্ষ"
    e.g. ১.৫ লক্ষ -> integer_part: "এক" fractional_part: "পাঁচ" quantity: "লক্ষ"

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
        -১২.৫০০৬ কোটি -> decimal { negative: "true" integer_part: "বারো"  fractional_part: "পাঁচ শূন্য শূন্য ছয়" quantity: "কোটি" }
        ১ কোটি -> decimal { integer_part: "এক" quantity: "কোটি" }
        12.34 -> decimal { integer_part: "বারো" fractional_part: "তিন চার" }
        -12.5006 -> decimal { negative: "true" integer_part: "বারো" fractional_part: "পাঁচ শূন্য শূন্য ছয়" }

    cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        # Get digit graphs from cardinal (maps Bengali digits to words)
        graph_digit = cardinal.digit | cardinal.zero
        cardinal_graph = cardinal.final_graph

        # Bengali digit sequence: Bengali digits → words with spaces
        bengali_digit_sequence = (graph_digit + pynini.closure(insert_space + graph_digit)).optimize()
        
        # Arabic digit sequence: Arabic digits → convert to Bengali → apply same sequence
        arabic_digit_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_digit_sequence = pynini.compose(
            arabic_digit_input,
            arabic_to_bengali_number @ bengali_digit_sequence,
        ).optimize()
        
        # Combined fractional part graph (supports both Bengali and Arabic digits)
        self.graph = (bengali_digit_sequence | arabic_digit_sequence).optimize()

        # Handle both "." and "," as decimal separators (common in Indian number systems)
        point = pynutil.delete(pynini.union(".", ","))

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + insert_space,
            0,
            1,
        )

        # Integer part uses cardinal_graph directly (already handles both script types)
        self.graph_fractional = pynutil.insert("fractional_part: \"") + self.graph + pynutil.insert("\"")
        self.graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")

        final_graph_wo_sign = self.graph_integer + point + insert_space + self.graph_fractional

        self.final_graph_wo_negative = final_graph_wo_sign | get_quantity(final_graph_wo_sign, cardinal_graph)

        final_graph = optional_graph_negative + self.final_graph_wo_negative

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
