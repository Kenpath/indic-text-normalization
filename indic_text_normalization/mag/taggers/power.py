# Copyright (c) 2026
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

import pynini
from pynini.lib import pynutil

from indic_text_normalization.mag.graph_utils import (
    GraphFst,
    NEMO_DIGIT,
    NEMO_MAG_DIGIT,
    NEMO_SUPERSCRIPT_DIGIT,
    NEMO_SUPERSCRIPT_MINUS,
    NEMO_SUPERSCRIPT_PLUS,
    superscript_to_digit,
    insert_space,
)


class PowerFst(GraphFst):
    """
    Classify powers/exponents with superscripts, e.g.
      "10⁻⁷" -> power { base: "दस" sign: "ऋणात्मक" exponent: "सात" }
      "2³"   -> power { base: "दुइ" exponent: "तीन" }
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="power", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph

        # Arabic digits -> Magadhi digits
        arabic_to_magadhi_digit = pynini.string_map(
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
        ).optimize()
        arabic_to_magadhi_number = pynini.closure(arabic_to_magadhi_digit).optimize()

        # Base number (Magadhi or Arabic)
        mag_base = pynini.compose(pynini.closure(NEMO_MAG_DIGIT, 1), cardinal_graph).optimize()
        ar_base = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1), arabic_to_magadhi_number @ cardinal_graph
        ).optimize()
        base_number = (mag_base | ar_base).optimize()

        # Optional sign
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

        # Superscript digits -> regular -> Magadhi -> cardinal
        superscript_number = pynini.closure(NEMO_SUPERSCRIPT_DIGIT, 1)
        exponent_value = pynini.compose(
            superscript_number,
            pynini.closure(superscript_to_digit) @ arabic_to_magadhi_number @ cardinal_graph,
        ).optimize()

        graph = (
            pynutil.insert('base: "')
            + base_number
            + pynutil.insert('"')
            + insert_space
            + optional_sign
            + pynutil.insert('exponent: "')
            + exponent_value
            + pynutil.insert('"')
        )

        self.fst = self.add_tokens(graph).optimize()

