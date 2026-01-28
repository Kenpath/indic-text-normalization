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

from indic_text_normalization.ml.graph_utils import GraphFst, NEMO_DIGIT, NEMO_HI_DIGIT, insert_space


class ScientificFst(GraphFst):
    """
    Classify ASCII scientific-notation-like strings.

    Examples:
      - "10.1-e5" -> scientific { mantissa: "പത്ത് ദശാംശം ഒന്ന്" exponent: "അഞ്ച്" }
      - "10.1e-5" -> scientific { mantissa: "പത്ത് ദശാംശം ഒന്ന്" sign: "നെഗറ്റീവ്" exponent: "അഞ്ച്" }
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="scientific", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph
        digit_word_graph = (cardinal.digit | cardinal.zero).optimize()

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

        # Integer part for mantissa
        ml_int = pynini.compose(pynini.closure(NEMO_HI_DIGIT, 1), cardinal_graph).optimize()
        ar_int = pynini.compose(pynini.closure(NEMO_DIGIT, 1), arabic_to_ml_number @ cardinal_graph).optimize()
        integer_graph = (ml_int | ar_int).optimize()

        # Fractional digits spoken digit-by-digit
        ml_frac = pynini.compose(
            pynini.closure(NEMO_HI_DIGIT, 1),
            digit_word_graph + pynini.closure(insert_space + digit_word_graph),
        ).optimize()
        ar_frac = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_ml_number @ (digit_word_graph + pynini.closure(insert_space + digit_word_graph)),
        ).optimize()
        fractional_graph = (ml_frac | ar_frac).optimize()

        point = pynutil.delete(".") + pynutil.insert(" ദശാംശം ")
        mantissa_graph = (integer_graph + point + fractional_graph).optimize()

        exponent_graph = integer_graph

        e_sep = pynini.closure(pynutil.delete("-"), 0, 1) + pynutil.delete(pynini.union("e", "E"))

        optional_sign = pynini.closure(
            pynutil.insert('sign: "')
            + (pynini.cross("-", "നെഗറ്റീവ്") | pynini.cross("+", "പോസിറ്റീവ്"))
            + pynutil.insert('"')
            + insert_space,
            0,
            1,
        )

        graph = (
            pynutil.insert('mantissa: "')
            + mantissa_graph
            + pynutil.insert('"')
            + insert_space
            + e_sep
            + optional_sign
            + pynutil.insert('exponent: "')
            + exponent_graph
            + pynutil.insert('"')
        )

        self.fst = self.add_tokens(graph).optimize()

