# Copyright (c) 2025
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

from indic_text_normalization.gu.graph_utils import GraphFst, NEMO_DIGIT, NEMO_GU_DIGIT, insert_space


class ScientificFst(GraphFst):
    """
    Classify ASCII scientific-notation-like strings.

    Supported examples (Gujarati):
      - "10.1-e5" -> scientific { mantissa: "દસ દશાંશ એક" exponent: "પાંચ" }
      - "10.1e-5" -> scientific { mantissa: "દસ દશાંશ એક" sign: "ઋણાત્મક" exponent: "પાંચ" }

    Verbalizer format:
      mantissa + " ગુણા દસ પાવર " + [sign] + exponent
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="scientific", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph
        digit_word_graph = (cardinal.digit | cardinal.zero).optimize()

        # Arabic digits -> Gujarati digits
        arabic_to_gujarati_digit = pynini.string_map(
            [
                ("0", "૦"),
                ("1", "૧"),
                ("2", "૨"),
                ("3", "૩"),
                ("4", "૪"),
                ("5", "૫"),
                ("6", "૬"),
                ("7", "૭"),
                ("8", "૮"),
                ("9", "૯"),
            ]
        ).optimize()
        arabic_to_gujarati_number = pynini.closure(arabic_to_gujarati_digit).optimize()

        # Integer part for mantissa
        gujarati_int = pynini.compose(pynini.closure(NEMO_GU_DIGIT, 1), cardinal_graph).optimize()
        arabic_int = pynini.compose(pynini.closure(NEMO_DIGIT, 1), arabic_to_gujarati_number @ cardinal_graph).optimize()
        integer_graph = (gujarati_int | arabic_int).optimize()

        # Fractional digits spoken digit-by-digit
        gujarati_frac = pynini.compose(
            pynini.closure(NEMO_GU_DIGIT, 1),
            digit_word_graph + pynini.closure(insert_space + digit_word_graph),
        ).optimize()
        arabic_frac = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_gujarati_number @ (digit_word_graph + pynini.closure(insert_space + digit_word_graph)),
        ).optimize()
        fractional_graph = (gujarati_frac | arabic_frac).optimize()

        # Decimal point in Gujarati
        point = pynutil.delete(".") + pynutil.insert(" દશાંશ ")
        mantissa_graph = (integer_graph + point + fractional_graph).optimize()

        # Exponent (integer)
        exponent_graph = integer_graph

        # e/E separator, optionally written as "-e" like "10.1-e5"
        e_sep = pynini.closure(pynutil.delete("-"), 0, 1) + pynutil.delete(pynini.union("e", "E"))

        optional_sign = pynini.closure(
            pynutil.insert('sign: "')
            + (pynini.cross("-", "ઋણાત્મક") | pynini.cross("+", "ધનાત્મક"))
            + pynutil.insert('"')
            + insert_space,
            0,
            1,
        )

        # Full scientific notation: mantissa + e/E + (optional sign) + exponent
        # Output: scientific { mantissa: "..." [sign: "..."] exponent: "..." }
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
