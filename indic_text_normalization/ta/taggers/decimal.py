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

from indic_text_normalization.ta.graph_utils import (
    GraphFst,
    NEMO_DIGIT,
    NEMO_TA_DIGIT,
    insert_space,
)
from indic_text_normalization.ta.utils import get_abs_path

quantities = pynini.string_file(get_abs_path("data/numbers/thousands.tsv"))

# Convert Arabic digits (0-9) to Tamil digits (௦-௯)
arabic_to_tamil_digit = pynini.string_map([
    ("0", "௦"), ("1", "௧"), ("2", "௨"), ("3", "௩"), ("4", "௪"),
    ("5", "௫"), ("6", "௬"), ("7", "௭"), ("8", "௮"), ("9", "௯")
]).optimize()
arabic_to_tamil_number = pynini.closure(arabic_to_tamil_digit).optimize()

# Create a graph that deletes commas from digit sequences (e.g., 1,000,001.50)
any_digit = pynini.union(NEMO_DIGIT, NEMO_TA_DIGIT)
delete_commas = (any_digit + pynini.closure(pynini.closure(pynutil.delete(","), 0, 1) + any_digit)).optimize()


def get_quantity(decimal: 'pynini.FstLike', cardinal_up_to_hundred: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. ௧ இலட்சம் -> integer_part: "ஒன்று" quantity: "இலட்சம்"
    e.g. ௧.௫ இலட்சம் -> integer_part: "ஒன்று" fractional_part: "ஐந்து" quantity: "இலட்சம்"

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
        -௧௨.௫௦௦௬ கோடி -> decimal { negative: "true" integer_part: "பன்னிரண்டு"  fractional_part: "ஐந்து பூஜ்யம் பூஜ்யம் ஆறு" quantity: "கோடி" }
        ௧ கோடி -> decimal { integer_part: "ஒன்று" quantity: "கோடி" }

    cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        graph_digit = cardinal.digit | cardinal.zero
        cardinal_graph = cardinal.final_graph

        tamil_digit_sequence = (graph_digit + pynini.closure(insert_space + graph_digit)).optimize()
        arabic_digit_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_digit_sequence = pynini.compose(
            arabic_digit_input,
            arabic_to_tamil_number @ tamil_digit_sequence,
        ).optimize()
        self.graph = (tamil_digit_sequence | arabic_digit_sequence).optimize()

        # Decimal separator
        point = pynutil.delete(".")

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + insert_space,
            0,
            1,
        )

        # Add comma support for integer parts
        integer_with_commas = pynini.compose(delete_commas, cardinal_graph).optimize()
        integer_graph = pynutil.add_weight(integer_with_commas, -0.1) | cardinal_graph

        self.graph_fractional = pynutil.insert("fractional_part: \"") + self.graph + pynutil.insert("\"")
        self.graph_integer = pynutil.insert("integer_part: \"") + integer_graph + pynutil.insert("\"")

        final_graph_wo_sign = self.graph_integer + point + insert_space + self.graph_fractional

        self.final_graph_wo_negative = final_graph_wo_sign | get_quantity(final_graph_wo_sign, integer_graph)

        final_graph = optional_graph_negative + self.final_graph_wo_negative

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

