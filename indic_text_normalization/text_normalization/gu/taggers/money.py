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

from indic_text_normalization.text_normalization.gu.graph_utils import (
    GraphFst,
    NEMO_DIGIT,
    NEMO_GU_DIGIT,
    insert_space,
)
from indic_text_normalization.text_normalization.gu.utils import get_abs_path

currency_graph = pynini.string_file(get_abs_path("data/money/currency.tsv"))

# Convert Arabic digits (0-9) to Gujarati digits (૦-૯)
arabic_to_gujarati_digit = pynini.string_map([
    ("0", "૦"), ("1", "૧"), ("2", "૨"), ("3", "૩"), ("4", "૪"),
    ("5", "૫"), ("6", "૬"), ("7", "૭"), ("8", "૮"), ("9", "૯")
]).optimize()
arabic_to_gujarati_number = pynini.closure(arabic_to_gujarati_digit).optimize()

# Gujarati suffixes that can follow money amounts
gujarati_suffixes = pynini.union("ના", "ની", "ને", "નો").optimize()


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, suppletive aware, e.g.
        ₹૫૦ -> money { money { currency_maj: "રૂપિયા" integer_part: "પચાસ" }
        ₹૫૦.૫૦ -> money { currency_maj: "રૂપિયા" integer_part: "પચાસ" fractional_part: "પચાસ" currency_min: "centiles" }
        ₹૦.૫૦ -> money { currency_maj: "રૂપિયા" integer_part: "શૂન્ય" fractional_part: "પચાસ" currency_min: "centiles" }
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
        
        # Accept both Gujarati digits and Arabic digits (convert Arabic to Gujarati)
        # Gujarati digits go directly to cardinal_graph, Arabic digits are converted first
        gujarati_digit_number = pynini.closure(NEMO_GU_DIGIT, 1).optimize()
        arabic_digit_number = pynini.closure(NEMO_DIGIT, 1).optimize()
        # Convert Arabic digits to Gujarati digits, then compose with cardinal_graph
        arabic_to_cardinal = pynini.compose(arabic_digit_number, arabic_to_gujarati_number @ cardinal_graph).optimize()
        # Gujarati digits go directly to cardinal_graph
        gujarati_to_cardinal = pynini.compose(gujarati_digit_number, cardinal_graph).optimize()
        # Combine both paths
        number_cardinal = arabic_to_cardinal | gujarati_to_cardinal
        
        integer = pynutil.insert('integer_part: "') + number_cardinal + pynutil.insert('"')
        fraction = pynutil.insert('fractional_part: "') + number_cardinal + pynutil.insert('"')
        currency_minor = pynutil.insert('currency_min: "') + pynutil.insert("centiles") + pynutil.insert('"')

        # Optional Gujarati suffixes after money amount
        optional_suffix = pynini.closure(pynutil.delete(gujarati_suffixes), 0, 1)

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

