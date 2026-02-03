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

from indic_text_normalization.sa.graph_utils import GraphFst, NEMO_DIGIT, NEMO_HI_DIGIT, insert_space


class ScientificFst(GraphFst):
    """
    Classify ASCII scientific-notation-like strings.

    Supported examples (Sanskrit):
      - "10.1-e5" -> scientific { mantissa: "दश दशमलव एकम्" exponent: "पञ्च" }
      - "10.1e-5" -> scientific { mantissa: "दश दशमलव एकम्" sign: "ऋणात्मक" exponent: "पञ्च" }

    Verbalizer format (see `sa/verbalizers/scientific.py`):
      mantissa + " गुणितम् दश घातः " + [sign] + exponent
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="scientific", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph
        digit_word_graph = (cardinal.digit | cardinal.zero).optimize()

        # Arabic digits -> Hindi/Sanskrit digits (Devanagari)
        arabic_to_hindi_digit = pynini.string_map(
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
        arabic_to_hindi_number = pynini.closure(arabic_to_hindi_digit).optimize()

        # Superscript mapping
        superscript_map = pynini.string_map([
            ("⁰", "0"), ("¹", "1"), ("²", "2"), ("³", "3"), ("⁴", "4"),
            ("⁵", "5"), ("⁶", "6"), ("⁷", "7"), ("⁸", "8"), ("⁹", "9"),
            ("⁻", "-")
        ])
        # Convert superscripts to normal ASCII for processing
        superscript_to_ascii = pynini.closure(superscript_map).optimize()

        # Integer part for mantissa
        hindi_int = pynini.compose(pynini.closure(NEMO_HI_DIGIT, 1), cardinal_graph).optimize()
        arabic_int = pynini.compose(pynini.closure(NEMO_DIGIT, 1), arabic_to_hindi_number @ cardinal_graph).optimize()
        integer_graph = (hindi_int | arabic_int).optimize()

        # Fractional digits spoken digit-by-digit
        hindi_frac = pynini.compose(
            pynini.closure(NEMO_HI_DIGIT, 1),
            digit_word_graph + pynini.closure(insert_space + digit_word_graph),
        ).optimize()
        arabic_frac = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_hindi_number @ (digit_word_graph + pynini.closure(insert_space + digit_word_graph)),
        ).optimize()
        fractional_graph = (hindi_frac | arabic_frac).optimize()

        # Sanskrit for decimal is usually "दशमलव" as well in this context
        point = pynutil.delete(".") + pynutil.insert(" दशमलव ")
        mantissa_graph = (integer_graph + point + fractional_graph).optimize()

        # Exponent (integer)
        exponent_graph = integer_graph

        # e/E separator, optionally written as "-e" like "10.1-e5"
        e_sep = pynini.closure(pynutil.delete("-"), 0, 1) + pynutil.delete(pynini.union("e", "E"))

        optional_sign = pynini.closure(
            pynutil.insert('sign: "')
            + (pynini.cross("-", "ऋणात्मक") | pynini.cross("+", "धनात्मक"))
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

        # Superscript path
        ascii_exponent_parser = (
             pynini.closure(
                pynutil.insert('sign: "')
                + (pynini.cross("-", "ऋणात्मक") | pynini.cross("+", "धनात्मक"))
                + pynutil.insert('"')
                + insert_space,
                0,
                1
             )
             + pynutil.insert('exponent: "')
             + (hindi_int | arabic_int)
             + pynutil.insert('"')
        )
        
        superscript_graph = pynini.compose(superscript_to_ascii, ascii_exponent_parser).optimize()

        graph_superscript = (
            pynutil.insert('mantissa: "')
            + integer_graph
            + pynutil.insert('"')
            + insert_space
            + superscript_graph
        )

        final_graph = graph | graph_superscript

        self.fst = self.add_tokens(final_graph).optimize()
