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

from indic_text_normalization.ml.graph_utils import (
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
    """
    Classify powers/exponents with superscripts, e.g.
      "10⁻⁷" -> power { base: "പത്ത്" sign: "നെഗറ്റീവ്" exponent: "ഏഴ്" }
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="power", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph

        # Arabic digits -> Malayalam digits
        arabic_to_ml_digit = pynini.string_map(
            [
                ("0", "൦"),
                ("1", "൧"),
                ("2", "൨"),
                ("3", "൩"),
                ("4", "൪"),
                ("5", "൫"),
                ("6", "൬"),
                ("7", "൭"),
                ("8", "൮"),
                ("9", "൯"),
            ]
        ).optimize()
        arabic_to_ml_number = pynini.closure(arabic_to_ml_digit).optimize()

        # Base number (Malayalam digits or Arabic)
        ml_base = pynini.compose(pynini.closure(NEMO_HI_DIGIT, 1), cardinal_graph).optimize()
        ar_base = pynini.compose(pynini.closure(NEMO_DIGIT, 1), arabic_to_ml_number @ cardinal_graph).optimize()
        base_number = (ml_base | ar_base).optimize()

        optional_sign = pynini.closure(
            pynutil.insert('sign: "')
            + (
                pynini.cross(NEMO_SUPERSCRIPT_MINUS, "നെഗറ്റീവ്")
                | pynini.cross(NEMO_SUPERSCRIPT_PLUS, "പോസിറ്റീവ്")
            )
            + pynutil.insert('"')
            + insert_space,
            0,
            1,
        )

        superscript_number = pynini.closure(NEMO_SUPERSCRIPT_DIGIT, 1)
        exponent_value = pynini.compose(
            superscript_number,
            pynini.closure(superscript_to_digit) @ arabic_to_ml_number @ cardinal_graph,
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

