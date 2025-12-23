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

from indic_text_normalization.text_normalization.ma.graph_utils import (
    NEMO_DIGIT,
    NEMO_HI_DIGIT,
    NEMO_SPACE,
    GraphFst,
)
from indic_text_normalization.text_normalization.ma.utils import get_abs_path

# Convert Arabic digits (0-9) to Malayalam digits (൦-൯)
arabic_to_hindi_digit = pynini.string_map([
    ("0", "൦"), ("1", "൧"), ("2", "൨"), ("3", "൩"), ("4", "൪"),
    ("5", "൫"), ("6", "൬"), ("7", "൭"), ("8", "൮"), ("9", "൯")
]).optimize()
arabic_to_hindi_number = pynini.closure(arabic_to_hindi_digit).optimize()


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
    "൨൩ ൪/൬" ->
    fraction { integer: "ഇരുപത്തിമൂന്ന്" numerator: "നാല്" denominator: "छः"}
    ൪/൬" ->
    fraction { numerator: "നാല്" denominator: "छः"}


    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph
        
        # Support both Malayalam and Arabic digits for integer, numerator, and denominator
        # Malayalam digits input
        hindi_number_input = pynini.closure(NEMO_HI_DIGIT, 1)
        hindi_number_graph = pynini.compose(hindi_number_input, cardinal_graph).optimize()
        
        # Arabic digits input
        arabic_number_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_number_graph = pynini.compose(
            arabic_number_input,
            arabic_to_hindi_number @ cardinal_graph
        ).optimize()
        
        # Combined number graph (supports both Malayalam and Arabic digits)
        number_graph = hindi_number_graph | arabic_number_graph

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + pynutil.insert(NEMO_SPACE), 0, 1
        )
        integer = pynutil.insert("integer_part: \"") + number_graph + pynutil.insert("\"")
        numerator = (
            pynutil.insert("numerator: \"")
            + number_graph
            + pynini.cross(pynini.union("/", NEMO_SPACE + "/" + NEMO_SPACE), "\" ")
        )
        denominator = pynutil.insert("denominator: \"") + number_graph + pynutil.insert("\"")

        graph = pynini.closure(integer + pynini.accep(NEMO_SPACE), 0, 1) + (numerator + denominator)
        graph = optional_graph_negative + graph

        self.graph = graph
        final_graph = self.add_tokens(self.graph)
        self.fst = final_graph.optimize()
