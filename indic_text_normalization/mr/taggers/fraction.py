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

from indic_text_normalization.mr.graph_utils import (
    MR_DIDH,
    MR_ADICH,
    MR_PAUNE,
    MR_SADHE,
    MR_SAVVA,
    NEMO_SPACE,
    GraphFst,
)
from indic_text_normalization.mr.utils import get_abs_path

MR_ONE_HALF = "१/२"  # 1/2
MR_ONE_QUARTER = "१/४"  # 1/4
MR_THREE_QUARTERS = "३/४"  # 3/4


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
    "२३ ४/६" ->
    fraction { integer: "तेवीस" numerator: "चार" denominator: "सहा"}
    ४/६" ->
    fraction { numerator: "चार" denominator: "सहा"}


    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph

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

        dedh_adich_graph = pynini.string_map(
            [("१" + NEMO_SPACE + MR_ONE_HALF, MR_DIDH), ("२" + NEMO_SPACE + MR_ONE_HALF, MR_ADICH)]
        )

        savva_numbers = cardinal_graph + pynini.cross(NEMO_SPACE + MR_ONE_QUARTER, "")
        savva_graph = pynutil.insert(MR_SAVVA) + pynutil.insert(NEMO_SPACE) + savva_numbers

        sadhe_numbers = cardinal_graph + pynini.cross(NEMO_SPACE + MR_ONE_HALF, "")
        sadhe_graph = pynutil.insert(MR_SADHE) + pynutil.insert(NEMO_SPACE) + sadhe_numbers

        paune = pynini.string_file(get_abs_path("data/whitelist/paune_mappings.tsv"))
        paune_numbers = paune + pynini.cross(NEMO_SPACE + MR_THREE_QUARTERS, "")
        paune_graph = pynutil.insert(MR_PAUNE) + pynutil.insert(NEMO_SPACE) + paune_numbers

        graph_dedh_adich = (
            pynutil.insert("morphosyntactic_features: \"")
            + dedh_adich_graph
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
            | pynutil.add_weight(graph_dedh_adich, -0.2)
            | pynutil.add_weight(graph_savva, -0.1)
            | pynutil.add_weight(graph_sadhe, -0.1)
            | pynutil.add_weight(graph_paune, -0.2)
        )

        self.graph = weighted_graph

        graph = self.graph
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()

