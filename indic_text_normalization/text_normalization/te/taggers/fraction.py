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

from indic_text_normalization.text_normalization.te.graph_utils import (
    TE_DEDH,
    TE_DHAI,
    TE_PAUNE,
    TE_SADHE,
    TE_SAVVA,
    NEMO_SPACE,
    GraphFst,
)
from indic_text_normalization.text_normalization.te.utils import get_abs_path

TE_ONE_HALF = "౧/౨"  # 1/2
TE_ONE_QUARTER = "౧/౪"  # 1/4
TE_THREE_QUARTERS = "౩/౪"  # 3/4


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
    "౨౩ ౪/౬" ->
    fraction { integer: "ఇరవై మూడు" numerator: "నాలుగు" denominator: "ఆరు"}
    ౪/౬" ->
    fraction { numerator: "నాలుగు" denominator: "ఆరు"}


    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        from indic_text_normalization.text_normalization.te.graph_utils import NEMO_DIGIT
        
        # Convert Arabic digits to Telugu for fractions
        arabic_to_telugu_digit = pynini.string_map([
            ("0", "౦"), ("1", "౧"), ("2", "౨"), ("3", "౩"), ("4", "౪"),
            ("5", "౫"), ("6", "౬"), ("7", "౭"), ("8", "౮"), ("9", "౯")
        ]).optimize()
        arabic_to_telugu_number = pynini.closure(arabic_to_telugu_digit).optimize()

        # Support both Telugu and Arabic digits
        telugu_cardinal_graph = cardinal.final_graph
        arabic_cardinal_graph = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_telugu_number @ telugu_cardinal_graph
        )
        cardinal_graph = telugu_cardinal_graph | arabic_cardinal_graph

        self.optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + pynutil.insert(NEMO_SPACE), 0, 1
        )
        self.integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        self.numerator = (
            pynutil.insert("numerator: \"")
            + cardinal_graph
            + pynini.cross(pynini.union("/", NEMO_SPACE + "/" + NEMO_SPACE), "\"")
            + pynutil.insert(NEMO_SPACE)
        )
        self.denominator = pynutil.insert("denominator: \"") + cardinal_graph + pynutil.insert("\"")

        dedh_dhai_graph = pynini.string_map(
            [("౧" + NEMO_SPACE + TE_ONE_HALF, TE_DEDH), ("౨" + NEMO_SPACE + TE_ONE_HALF, TE_DHAI)]
        )

        savva_numbers = cardinal_graph + pynini.cross(NEMO_SPACE + TE_ONE_QUARTER, "")
        savva_graph = pynutil.insert(TE_SAVVA) + pynutil.insert(NEMO_SPACE) + savva_numbers

        sadhe_numbers = cardinal_graph + pynini.cross(NEMO_SPACE + TE_ONE_HALF, "")
        sadhe_graph = pynutil.insert(TE_SADHE) + pynutil.insert(NEMO_SPACE) + sadhe_numbers

        paune = pynini.string_file(get_abs_path("data/whitelist/paune_mappings.tsv"))
        paune_numbers = paune + pynini.cross(NEMO_SPACE + TE_THREE_QUARTERS, "")
        paune_graph = pynutil.insert(TE_PAUNE) + pynutil.insert(NEMO_SPACE) + paune_numbers

        graph_dedh_dhai = (
            pynutil.insert("morphosyntactic_features: \"")
            + dedh_dhai_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        graph_savva = (
            pynutil.insert("morphosyntactic_features: \"")
            + savva_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        graph_sadhe = (
            pynutil.insert("morphosyntactic_features: \"")
            + sadhe_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        graph_paune = (
            pynutil.insert("morphosyntactic_features: \"")
            + paune_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        final_graph = (
            self.optional_graph_negative
            + pynini.closure(self.integer + pynini.accep(NEMO_SPACE), 0, 1)
            + self.numerator
            + self.denominator
        )

        weighted_graph = (
            final_graph
            | pynutil.add_weight(graph_dedh_dhai, -0.2)
            | pynutil.add_weight(graph_savva, -0.1)
            | pynutil.add_weight(graph_sadhe, -0.1)
            | pynutil.add_weight(graph_paune, -0.2)
        )

        self.graph = weighted_graph

        graph = self.graph
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()

