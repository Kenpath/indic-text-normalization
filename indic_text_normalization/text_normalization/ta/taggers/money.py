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

from indic_text_normalization.text_normalization.ta.graph_utils import (
    GraphFst,
    NEMO_DIGIT,
    NEMO_TA_DIGIT,
    insert_space,
)
from indic_text_normalization.text_normalization.ta.utils import get_abs_path

currency_graph = pynini.string_file(get_abs_path("data/money/currency.tsv"))

# Convert Arabic digits (0-9) to Tamil digits (௦-௯)
arabic_to_tamil_digit = pynini.string_map([
    ("0", "௦"), ("1", "௧"), ("2", "௨"), ("3", "௩"), ("4", "௪"),
    ("5", "௫"), ("6", "௬"), ("7", "௭"), ("8", "௮"), ("9", "௯")
]).optimize()
arabic_to_tamil_number = pynini.closure(arabic_to_tamil_digit).optimize()

# Tamil suffixes that can follow money amounts
tamil_suffixes = pynini.union("க்கு", "க்கு", "க்கு", "க்கு").optimize()


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, suppletive aware, e.g.
        ₹௫௦ -> money { money { currency_maj: "ரூபாய்" integer_part: "ஐம்பது" }
        ₹௫௦.௫௦ -> money { currency_maj: "ரூபாய்" integer_part: "ஐம்பது" fractional_part: "ஐம்பது" currency_min: "centiles" }
        ₹௦.௫௦ -> money { currency_maj: "ரூபாய்" integer_part: "பூஜ்யம்" fractional_part: "ஐம்பது" currency_min: "centiles" }
    Note that the 'centiles' string is a placeholder to handle by the verbalizer by applying the corresponding minor currency denomination

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
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
        
        # Accept both Tamil digits and Arabic digits (convert Arabic to Tamil)
        # Tamil digits go directly to cardinal_graph, Arabic digits are converted first
        tamil_digit_number = pynini.closure(NEMO_TA_DIGIT, 1).optimize()
        arabic_digit_number = pynini.closure(NEMO_DIGIT, 1).optimize()
        # Convert Arabic digits to Tamil digits, then compose with cardinal_graph
        arabic_to_cardinal = pynini.compose(arabic_digit_number, arabic_to_tamil_number @ cardinal_graph).optimize()
        # Tamil digits go directly to cardinal_graph
        tamil_to_cardinal = pynini.compose(tamil_digit_number, cardinal_graph).optimize()
        # Combine both paths
        number_cardinal = arabic_to_cardinal | tamil_to_cardinal
        
        integer = pynutil.insert('integer_part: "') + number_cardinal + pynutil.insert('"')
        fraction = pynutil.insert('fractional_part: "') + number_cardinal + pynutil.insert('"')
        currency_minor = pynutil.insert('currency_min: "') + pynutil.insert("centiles") + pynutil.insert('"')

        # Optional Tamil suffixes after money amount
        optional_suffix = pynini.closure(pynutil.delete(tamil_suffixes), 0, 1)

        graph_major_only = optional_graph_negative + currency_major + insert_space + integer + optional_suffix
        graph_major_and_minor = (
            optional_graph_negative
            + currency_major
            + insert_space
            + integer
            + pynini.cross(".", " ")
            + fraction
            + insert_space
            + currency_minor
            + optional_suffix
        )

        graph_currencies = graph_major_only | graph_major_and_minor

        graph = graph_currencies.optimize()
        final_graph = self.add_tokens(graph)
        self.fst = final_graph

