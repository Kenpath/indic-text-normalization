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

from indic_text_normalization.pa.graph_utils import (
    GraphFst,
    NEMO_DIGIT,
    NEMO_HI_DIGIT,
    NEMO_SUPERSCRIPT_DIGIT,
    NEMO_SUPERSCRIPT_MINUS,
    NEMO_SUPERSCRIPT_PLUS,
    superscript_to_digit,
    insert_space,
)


class PowerFst(GraphFst):
    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="power", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph

        hindi_base_input = pynini.closure(NEMO_HI_DIGIT, 1)
        hindi_base = pynini.compose(hindi_base_input, cardinal_graph).optimize()

        arabic_base_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_to_hindi = pynini.closure(
            pynini.string_map(
                [
                    ("0", "०"),
                    ("1", "१"),
                    ("2", "२"),
                    ("3", "३"),
                    ("4", "४"),
                    ("5", "५"),
                    ("6", "६"),
                    ("7", "७"),
                    ("8", "८"),
                    ("9", "९"),
                ]
            )
        ).optimize()
        arabic_base = pynini.compose(arabic_base_input, arabic_to_hindi @ cardinal_graph).optimize()
        base_number = hindi_base | arabic_base

        optional_sign = pynini.closure(
            pynutil.insert('sign: "')
            + (
                pynini.cross(NEMO_SUPERSCRIPT_MINUS, "ऋणात्मक")
                | pynini.cross(NEMO_SUPERSCRIPT_PLUS, "धनात्मक")
            )
            + pynutil.insert('"')
            + insert_space,
            0,
            1,
        )

        superscript_number = pynini.closure(NEMO_SUPERSCRIPT_DIGIT, 1)
        exponent_value = pynini.compose(
            superscript_number,
            pynini.closure(superscript_to_digit) @ arabic_to_hindi @ cardinal_graph,
        ).optimize()

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

        self.fst = self.add_tokens(power_expr).optimize()
