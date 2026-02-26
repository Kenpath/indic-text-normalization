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

from indic_text_normalization.kn.graph_utils import GraphFst, insert_space
from indic_text_normalization.kn.utils import get_abs_path

currency_graph = pynini.string_file(get_abs_path("data/money/currency.tsv"))


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, suppletive aware, e.g.
        ₹೫೦ -> money { money { currency_maj: "ರೂಪಾಯಿ" integer_part: "ಐವತ್ತು" }
        ₹೫೦.೫೦ -> money { currency_maj: "ರೂಪಾಯಿ" integer_part: "ಐವತ್ತು" fractional_part: "ಐವತ್ತು" currency_min: "centiles" }
        ₹೦.೫೦ -> money { currency_maj: "ರೂಪಾಯಿ" integer_part: "ಸೊನ್ನೆ" fractional_part: "ಐವತ್ತು" currency_min: "centiles" }
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
        cardinal_with_commas = cardinal_graph

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + insert_space,
            0,
            1,
        )
        currency_major = pynutil.insert('currency_maj: "') + currency_graph + pynutil.insert('"')
        optional_space = pynini.closure(pynini.accep(" "), 0, 1)
        integer = (
            pynutil.insert('integer_part: "')
            + (pynutil.add_weight(cardinal_with_commas, -0.1) | cardinal_graph)
            + pynutil.insert('"')
        )
        fraction = (
            pynutil.insert('fractional_part: "')
            + (pynutil.add_weight(cardinal_with_commas, -0.1) | cardinal_graph)
            + pynutil.insert('"')
        )
        currency_minor = pynutil.insert('currency_min: "') + pynutil.insert("centiles") + pynutil.insert('"')

        optional_slash_dash = pynini.closure(
            pynutil.add_weight(pynini.closure(pynini.accep(" "), 0, 1) + pynutil.delete("/-"), -0.1),
            0,
            1,
        )

        graph_major_only = optional_graph_negative + currency_major + optional_space + insert_space + integer + optional_slash_dash
        graph_major_and_minor = (
            optional_graph_negative
            + currency_major
            + optional_space
            + insert_space
            + integer
            + optional_space
            + pynini.cross(".", " ")
            + fraction
            + insert_space
            + currency_minor
            + optional_slash_dash
        )

        graph_major_only_suffix = optional_graph_negative + integer + insert_space + optional_space + currency_major + optional_slash_dash
        graph_major_and_minor_suffix = (
            optional_graph_negative
            + integer
            + optional_space
            + pynini.cross(".", " ")
            + fraction
            + optional_space
            + insert_space
            + currency_minor
            + insert_space
            + currency_major
            + optional_slash_dash
        )

        graph_currencies = (
            graph_major_only
            | graph_major_and_minor
            | pynutil.add_weight(graph_major_only_suffix | graph_major_and_minor_suffix, 0.5)
        )

        graph = graph_currencies.optimize()
        final_graph = self.add_tokens(graph)
        self.fst = final_graph
