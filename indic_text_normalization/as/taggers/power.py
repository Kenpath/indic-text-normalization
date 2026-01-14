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

from ..graph_utils import (
    GraphFst,
    NEMO_DIGIT,
    NEMO_AS_DIGIT,
    NEMO_SUPERSCRIPT_DIGIT,
    NEMO_SUPERSCRIPT_MINUS,
    NEMO_SUPERSCRIPT_PLUS,
    superscript_to_digit,
    superscript_to_sign,
    insert_space,
)


class PowerFst(GraphFst):
    """
    Finite state transducer for classifying powers/exponents with superscripts, e.g.
        "10⁻⁷" -> power { base: "দহ" sign: "ঋণাত্মক" exponent: "সাত" }
        "2³" -> power { base: "দুই" exponent: "তিন" }

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="power", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph

        # Base number (regular digits - Assamese or Arabic)
        assamese_base_input = pynini.closure(NEMO_AS_DIGIT, 1)
        assamese_base = pynini.compose(assamese_base_input, cardinal_graph).optimize()
        
        arabic_base_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_to_assamese = pynini.closure(
            pynini.string_map([
                ("0", "০"), ("1", "১"), ("2", "২"), ("3", "৩"), ("4", "৪"),
                ("5", "৫"), ("6", "৬"), ("7", "৭"), ("8", "৮"), ("9", "৯")
            ])
        ).optimize()
        arabic_base = pynini.compose(arabic_base_input, arabic_to_assamese @ cardinal_graph).optimize()
        
        base_number = assamese_base | arabic_base

        # Superscript exponent
        # Optional sign
        optional_sign = pynini.closure(
            pynutil.insert('sign: "')
            + (
                pynini.cross(NEMO_SUPERSCRIPT_MINUS, "ঋণাত্মক")
                | pynini.cross(NEMO_SUPERSCRIPT_PLUS, "ধনাত্মক")
            )
            + pynutil.insert('"')
            + insert_space,
            0,
            1,
        )

        # Superscript digits -> convert to regular -> cardinal
        superscript_number = pynini.closure(NEMO_SUPERSCRIPT_DIGIT, 1)
        exponent_value = pynini.compose(
            superscript_number,
            pynini.closure(superscript_to_digit) @ arabic_to_assamese @ cardinal_graph
        ).optimize()

        # Complete power expression
        power_expr = (
            pynutil.insert('base: "')
            + base_number
            + pynutil.insert('"')
            + insert_space
            + optional_sign
            + pynutil.insert('exponent: "')
            + exponent_value
            + pynutil.insert('"')
        )

        final_graph = self.add_tokens(power_expr)
        self.fst = final_graph.optimize()
