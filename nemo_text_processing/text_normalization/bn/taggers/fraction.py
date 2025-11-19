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

from nemo_text_processing.text_normalization.bn.graph_utils import (
    NEMO_DIGIT,
    NEMO_SPACE,
    BN_DEDH,
    BN_DHAI,
    BN_PAUNE,
    BN_SADHE,
    BN_SAVVA,
    GraphFst,
)
from nemo_text_processing.text_normalization.bn.utils import get_abs_path

# Convert Arabic digits (0-9) to Bengali digits (০-৯)
arabic_to_bengali_digit = pynini.string_map([
    ("0", "০"), ("1", "১"), ("2", "২"), ("3", "৩"), ("4", "৪"),
    ("5", "৫"), ("6", "৬"), ("7", "৭"), ("8", "৮"), ("9", "৯")
]).optimize()
arabic_to_bengali_number = pynini.closure(arabic_to_bengali_digit).optimize()

# Bengali fraction constants
BN_ONE_HALF = "১/২"  # 1/2
BN_ONE_QUARTER = "১/৪"  # 1/4
BN_THREE_QUARTERS = "৩/৪"  # 3/4


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
    "২৩ ৪/৬" ->
    fraction { integer: "তেইশ" numerator: "চার" denominator: "ছয়"}
    ৪/৬" ->
    fraction { numerator: "চার" denominator: "ছয়"}


    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph

        # Support both Bengali and Arabic digits
        bengali_cardinal_graph = cardinal_graph
        arabic_cardinal_graph = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_bengali_number @ cardinal_graph
        ).optimize()
        combined_cardinal_graph = bengali_cardinal_graph | arabic_cardinal_graph

        self.optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + pynutil.insert(NEMO_SPACE), 0, 1
        )
        self.integer = pynutil.insert("integer_part: \"") + combined_cardinal_graph + pynutil.insert("\"")
        self.numerator = (
            pynutil.insert("numerator: \"")
            + combined_cardinal_graph
            + pynini.cross(pynini.union("/", NEMO_SPACE + "/" + NEMO_SPACE), "\"")
            + pynutil.insert(NEMO_SPACE)
        )
        self.denominator = pynutil.insert("denominator: \"") + combined_cardinal_graph + pynutil.insert("\"")

        # Handle special Bengali fraction words
        # দেড় (dedh) = 1.5 or 1 1/2
        dedh_dhai_graph = pynini.string_map([
            ("১" + NEMO_SPACE + BN_ONE_HALF, BN_DEDH), 
            ("২" + NEMO_SPACE + BN_ONE_HALF, BN_DHAI),
            # Also support Arabic digits
            ("1" + NEMO_SPACE + "1/2", BN_DEDH),
            ("2" + NEMO_SPACE + "1/2", BN_DHAI)
        ])

        # সাড়ে (savva) = quarter more (1.25 or 1 1/4)
        savva_numbers = combined_cardinal_graph + pynini.cross(NEMO_SPACE + BN_ONE_QUARTER, "")
        savva_numbers |= combined_cardinal_graph + pynini.cross(NEMO_SPACE + "1/4", "")
        savva_graph = pynutil.insert(BN_SAVVA) + pynutil.insert(NEMO_SPACE) + savva_numbers

        # সাড়ে (sadhe) = half more (X.5 or X 1/2)
        sadhe_numbers = combined_cardinal_graph + pynini.cross(NEMO_SPACE + BN_ONE_HALF, "")
        sadhe_numbers |= combined_cardinal_graph + pynini.cross(NEMO_SPACE + "1/2", "")
        sadhe_graph = pynutil.insert(BN_SADHE) + pynutil.insert(NEMO_SPACE) + sadhe_numbers

        # পৌনে (paune) = three quarters (3/4)
        paune = pynini.string_file(get_abs_path("data/whitelist/paune_mappings.tsv"))
        # Support both Bengali and Arabic digits for paune
        bengali_paune_numbers = paune + pynini.cross(NEMO_SPACE + BN_THREE_QUARTERS, "")
        arabic_paune_numbers = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_bengali_number @ paune
        ) + pynini.cross(NEMO_SPACE + "3/4", "")
        paune_numbers = bengali_paune_numbers | arabic_paune_numbers
        paune_graph = pynutil.insert(BN_PAUNE) + pynutil.insert(NEMO_SPACE) + paune_numbers

        # Regular fraction graph
        final_graph = (
            self.optional_graph_negative
            + pynini.closure(self.integer + pynini.accep(NEMO_SPACE), 0, 1)
            + self.numerator
            + self.denominator
        )

        # Special fraction graphs with weights (higher priority)
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

        # Combine all graphs with weights (special cases have higher priority)
        weighted_graph = (
            final_graph
            | pynutil.add_weight(graph_dedh_dhai, -0.2)
            | pynutil.add_weight(graph_savva, -0.1)
            | pynutil.add_weight(graph_sadhe, -0.1)
            | pynutil.add_weight(graph_paune, -0.2)
        )

        graph = weighted_graph
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()

