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

from nemo_text_processing.text_normalization.bho.graph_utils import (
    BHO_DEDH,
    BHO_DHAI,
    BHO_PAUNE,
    BHO_SADHE,
    BHO_SAVVA,
    NEMO_SPACE,
    NEMO_BHO_DIGIT,
    NEMO_DIGIT,
    GraphFst,
)
from nemo_text_processing.text_normalization.bho.utils import get_abs_path

# Bhojpuri fraction constants using Devanagari digits
BHO_ONE_HALF = "१/२"  # 1/2
BHO_ONE_QUARTER = "१/४"  # 1/4
BHO_THREE_QUARTERS = "३/४"  # 3/4


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
    "२३ ४/६" ->
    fraction { integer: "तेईस" numerator: "चार" denominator: "छः"}
    ४/६" ->
    fraction { numerator: "चार" denominator: "छः"}


    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        # Convert Arabic digits to Bhojpuri (Devanagari) digits for fractions
        arabic_to_bhojpuri_digit = pynini.string_map([
            ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
            ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९")
        ]).optimize()
        arabic_to_bhojpuri_number = pynini.closure(arabic_to_bhojpuri_digit).optimize()

        # Support both Bhojpuri (Devanagari) and Arabic digits
        cardinal_graph = cardinal.final_graph
        
        # Handle Arabic digits by converting to Bhojpuri first
        arabic_cardinal_graph = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_bhojpuri_number @ cardinal_graph
        )
        
        # Combine both Bhojpuri and Arabic digit support
        final_cardinal_graph = cardinal_graph | arabic_cardinal_graph

        self.optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + pynutil.insert(NEMO_SPACE), 0, 1
        )
        self.integer = pynutil.insert("integer_part: \"") + final_cardinal_graph + pynutil.insert("\"")
        self.numerator = (
            pynutil.insert("numerator: \"")
            + final_cardinal_graph
            + pynini.cross(pynini.union("/", NEMO_SPACE + "/" + NEMO_SPACE), "\"")
            + pynutil.insert(NEMO_SPACE)
        )
        self.denominator = pynutil.insert("denominator: \"") + final_cardinal_graph + pynutil.insert("\"")

        # Special fraction patterns: डेढ़ (1.5), ढाई (2.5)
        dedh_dhai_graph = pynini.string_map(
            [("१" + NEMO_SPACE + BHO_ONE_HALF, BHO_DEDH), ("२" + NEMO_SPACE + BHO_ONE_HALF, BHO_DHAI)]
        )

        # सवा (quarter more, 1.25)
        savva_numbers = final_cardinal_graph + pynini.cross(NEMO_SPACE + BHO_ONE_QUARTER, "")
        savva_graph = pynutil.insert(BHO_SAVVA) + pynutil.insert(NEMO_SPACE) + savva_numbers

        # साढ़े (half more, X.5)
        sadhe_numbers = final_cardinal_graph + pynini.cross(NEMO_SPACE + BHO_ONE_HALF, "")
        sadhe_graph = pynutil.insert(BHO_SADHE) + pynutil.insert(NEMO_SPACE) + sadhe_numbers

        # पौने (quarter less, 0.75)
        paune = pynini.string_file(get_abs_path("data/whitelist/paune_mappings.tsv"))
        paune_numbers = paune + pynini.cross(NEMO_SPACE + BHO_THREE_QUARTERS, "")
        paune_graph = pynutil.insert(BHO_PAUNE) + pynutil.insert(NEMO_SPACE) + paune_numbers

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

