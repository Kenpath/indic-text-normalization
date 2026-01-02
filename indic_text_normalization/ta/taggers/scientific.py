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

from indic_text_normalization.ta.graph_utils import GraphFst, NEMO_DIGIT, NEMO_TA_DIGIT, insert_space


class ScientificFst(GraphFst):
    """
    Classify ASCII scientific-notation-like strings.

    Supported examples (Tamil):
      - "10.1-e5" -> scientific { mantissa: "<spoken>" exponent: "<spoken>" }
      - "10.1e-5" -> scientific { mantissa: "<spoken>" sign: "மைனஸ்" exponent: "<spoken>" }

    Verbalizer format (see `ta/verbalizers/scientific.py`):
      mantissa + " பெருக்கல் பத்து பவர் " + [sign] + exponent
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="scientific", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph
        digit_word_graph = (cardinal.digit | cardinal.zero).optimize()

        # Arabic digits -> Tamil digits
        arabic_to_tamil_digit = pynini.string_map(
            [
                ("0", "௦"),
                ("1", "௧"),
                ("2", "௨"),
                ("3", "௩"),
                ("4", "௪"),
                ("5", "௫"),
                ("6", "௬"),
                ("7", "௭"),
                ("8", "௮"),
                ("9", "௯"),
            ]
        ).optimize()
        arabic_to_tamil_number = pynini.closure(arabic_to_tamil_digit).optimize()

        # Integer part for mantissa
        tamil_int = pynini.compose(pynini.closure(NEMO_TA_DIGIT, 1), cardinal_graph).optimize()
        arabic_int = pynini.compose(pynini.closure(NEMO_DIGIT, 1), arabic_to_tamil_number @ cardinal_graph).optimize()
        integer_graph = (tamil_int | arabic_int).optimize()

        # Fractional digits spoken digit-by-digit
        tamil_frac = pynini.compose(
            pynini.closure(NEMO_TA_DIGIT, 1),
            digit_word_graph + pynini.closure(insert_space + digit_word_graph),
        ).optimize()
        arabic_frac = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_tamil_number @ (digit_word_graph + pynini.closure(insert_space + digit_word_graph)),
        ).optimize()
        fractional_graph = (tamil_frac | arabic_frac).optimize()

        point = pynutil.delete(".") + pynutil.insert(" தசமம் ")
        mantissa_graph = (integer_graph + point + fractional_graph).optimize()

        exponent_graph = integer_graph

        # e/E separator, optionally written as "-e" like "10.1-e5"
        e_sep = pynini.closure(pynutil.delete("-"), 0, 1) + pynutil.delete(pynini.union("e", "E"))

        optional_sign = pynini.closure(
            pynutil.insert('sign: "')
            + (pynini.cross("-", "மைனஸ்") | pynini.cross("+", "பிளஸ்"))
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


