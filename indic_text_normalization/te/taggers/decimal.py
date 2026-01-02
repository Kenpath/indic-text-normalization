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

from indic_text_normalization.te.graph_utils import GraphFst, NEMO_DIGIT, NEMO_TE_DIGIT, insert_space
from indic_text_normalization.te.utils import get_abs_path

quantities = pynini.string_file(get_abs_path("data/numbers/thousands.tsv"))

# Create a graph that deletes commas from digit sequences
# This handles Indian number format where commas are separators (e.g., 1,000,001.50)
any_digit = pynini.union(NEMO_DIGIT, NEMO_TE_DIGIT)
# Pattern: digit (comma? digit)* - accepts digits with optional commas, deletes commas
delete_commas = (
    any_digit
    + pynini.closure(pynini.closure(pynutil.delete(","), 0, 1) + any_digit)
).optimize()


def get_quantity(decimal: 'pynini.FstLike', cardinal_up_to_hundred: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. ౧ లక్షం -> integer_part: "ఒకటి" quantity: "లక్షం"
    e.g. ౧.౫ లక్షం -> integer_part: "ఒకటి" fractional_part: "ఐదు" quantity: "లక్షం"

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
        -౧౨.౫౦౦౬ శతకోటి -> decimal { negative: "true" integer_part: "పన్నెండు"  fractional_part: "ఐదు సున్నా సున్నా ఆరు" quantity: "శతకోటి" }
        ౧ శతకోటి -> decimal { integer_part: "ఒకటి" quantity: "శతకోటి" }

    cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        # Convert Arabic digits to Telugu for decimal parts
        arabic_to_telugu_digit = pynini.string_map([
            ("0", "౦"), ("1", "౧"), ("2", "౨"), ("3", "౩"), ("4", "౪"),
            ("5", "౫"), ("6", "౬"), ("7", "౭"), ("8", "౮"), ("9", "౯")
        ]).optimize()
        arabic_to_telugu_number = pynini.closure(arabic_to_telugu_digit).optimize()

        graph_digit = cardinal.digit | cardinal.zero
        cardinal_graph = cardinal.final_graph

        # Support both Telugu and Arabic digits for fractional part
        telugu_fractional = graph_digit + pynini.closure(insert_space + graph_digit)
        arabic_fractional = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1) + pynini.closure(NEMO_DIGIT),
            arabic_to_telugu_number @ (graph_digit + pynini.closure(insert_space + graph_digit))
        )
        self.graph = (telugu_fractional | arabic_fractional).optimize()

        point = pynutil.delete(".")

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + insert_space,
            0,
            1,
        )

        # Add comma support for integer parts
        # Telugu with commas
        telugu_integer_with_commas = pynini.compose(delete_commas, cardinal_graph).optimize()
        telugu_integer_combined = pynutil.add_weight(telugu_integer_with_commas, -0.1) | cardinal_graph

        # Arabic digits need conversion
        arabic_digit_input = pynini.closure(NEMO_DIGIT, 1)

        # Arabic with commas
        arabic_integer_with_commas = pynini.compose(
            delete_commas,
            arabic_digit_input @ cardinal_graph
        ).optimize()

        # Regular Arabic digits
        arabic_integer_graph = arabic_digit_input @ cardinal_graph

        # Combined Arabic graph
        arabic_integer_combined = pynutil.add_weight(arabic_integer_with_commas, -0.1) | arabic_integer_graph

        # Combined integer graph (supports both Telugu and Arabic digits, with and without commas)
        integer_graph = telugu_integer_combined | arabic_integer_combined

        self.graph_fractional = pynutil.insert("fractional_part: \"") + self.graph + pynutil.insert("\"")
        self.graph_integer = pynutil.insert("integer_part: \"") + integer_graph + pynutil.insert("\"")

        final_graph_wo_sign = self.graph_integer + point + insert_space + self.graph_fractional

        self.final_graph_wo_negative = final_graph_wo_sign | get_quantity(final_graph_wo_sign, cardinal_graph)

        final_graph = optional_graph_negative + self.final_graph_wo_negative

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

