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

from indic_text_normalization.text_normalization.mag.graph_utils import (
    GraphFst,
    NEMO_DIGIT,
    NEMO_MAG_DIGIT,
    insert_space,
)
from indic_text_normalization.text_normalization.mag.utils import get_abs_path

currency_graph = pynini.string_file(get_abs_path("data/money/currency.tsv"))

# Convert Arabic digits (0-9) to Magadhi digits (०-९)
arabic_to_magadhi_digit = pynini.string_map([
    ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
    ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९")
]).optimize()
arabic_to_magadhi_number = pynini.closure(arabic_to_magadhi_digit).optimize()


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, suppletive aware, e.g.
        ₹५० -> money { currency_maj: "रुपये" integer_part: "पचास" }
        ₹५०.५० -> money { currency_maj: "रुपये" integer_part: "पचास" fractional_part: "पचास" currency_min: "centiles" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="money", kind="classify")

        cardinal_graph = cardinal.final_graph

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + insert_space,
            0,
            1,
        )
        currency_major = pynutil.insert('currency_maj: "') + currency_graph + pynutil.insert('"')
        
        # Accept both Magadhi digits and Arabic digits (convert Arabic to Magadhi)
        magadhi_digit_number = pynini.closure(NEMO_MAG_DIGIT, 1).optimize()
        arabic_digit_number = pynini.closure(NEMO_DIGIT, 1).optimize()
        # Convert Arabic digits to Magadhi digits, then compose with cardinal_graph
        arabic_to_cardinal = pynini.compose(arabic_digit_number, arabic_to_magadhi_number @ cardinal_graph).optimize()
        # Magadhi digits go directly to cardinal_graph
        magadhi_to_cardinal = pynini.compose(magadhi_digit_number, cardinal_graph).optimize()
        # Combine both paths
        number_cardinal = arabic_to_cardinal | magadhi_to_cardinal
        
        integer = pynutil.insert('integer_part: "') + number_cardinal + pynutil.insert('"')
        fraction = pynutil.insert('fractional_part: "') + number_cardinal + pynutil.insert('"')
        currency_minor = pynutil.insert('currency_min: "') + pynutil.insert("centiles") + pynutil.insert('"')

        graph_major_only = optional_graph_negative + currency_major + insert_space + integer
        graph_major_and_minor = (
            optional_graph_negative
            + currency_major
            + insert_space
            + integer
            + pynini.cross(".", " ")
            + fraction
            + insert_space
            + currency_minor
        )

        graph_currencies = graph_major_only | graph_major_and_minor

        graph = graph_currencies.optimize()
        final_graph = self.add_tokens(graph)
        self.fst = final_graph

