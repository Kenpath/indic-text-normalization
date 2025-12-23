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

from indic_text_normalization.text_normalization.do.graph_utils import (
    GraphFst,
    NEMO_DIGIT,
    NEMO_HI_DIGIT,
    insert_space,
)
from indic_text_normalization.text_normalization.do.utils import get_abs_path

currency_graph = pynini.string_file(get_abs_path("data/money/currency.tsv"))


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, suppletive aware, e.g.
        ₹५० -> money { money { currency_maj: "रुपए" integer_part: "पचास" }
        ₹५०.५० -> money { currency_maj: "रुपए" integer_part: "पचास" fractional_part: "पचास" currency_min: "centiles" }
        ₹०.५० -> money { currency_maj: "रुपए" integer_part: "शून्य" fractional_part: "पचास" currency_min: "centiles" }
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

        # Create a graph that deletes commas from digit sequences
        # This handles Indian number format where commas are separators (e.g., 5,67,300)
        any_digit = pynini.union(NEMO_DIGIT, NEMO_HI_DIGIT)
        # Pattern: digit (comma? digit)* - accepts digits with optional commas, deletes commas
        # This creates a transducer: input (with commas) -> output (without commas)
        delete_commas = (
            any_digit
            + pynini.closure(pynini.closure(pynutil.delete(","), 0, 1) + any_digit)
        ).optimize()
        # Compose: numbers with commas -> delete commas -> cardinal conversion
        # The composition works because delete_commas outputs digits, which cardinal_graph accepts as input
        cardinal_with_commas = pynini.compose(delete_commas, cardinal_graph).optimize()

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + insert_space,
            0,
            1,
        )
        currency_major = pynutil.insert('currency_maj: "') + currency_graph + pynutil.insert('"')
        # Use cardinal_with_commas (higher priority) to handle numbers with commas, fallback to regular cardinal_graph
        integer = pynutil.insert('integer_part: "') + (pynutil.add_weight(cardinal_with_commas, -0.1) | cardinal_graph) + pynutil.insert('"')
        fraction = pynutil.insert('fractional_part: "') + (pynutil.add_weight(cardinal_with_commas, -0.1) | cardinal_graph) + pynutil.insert('"')
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
