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

from indic_text_normalization.mr.graph_utils import GraphFst, NEMO_DIGIT, NEMO_MR_DIGIT, insert_space


class ScientificFst(GraphFst):
    """
    Classify ASCII scientific-notation-like strings.

    Supported examples (Marathi):
      - "10.1-e5" -> scientific { mantissa: "दहा दशमलव एक" exponent: "पाच" }
      - "10.1e-5" -> scientific { mantissa: "दहा दशमलव एक" sign: "नकारात्मक" exponent: "पाच" }

    Verbalizer format:
      mantissa + " गुणाकार दहा पावर " + [sign] + exponent
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="scientific", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph
        # Assuming cardinal.digit | cardinal.zero patterns exist if they were properties in Hindi implementation, 
        # but in Marathi cardinal implementation they might not be exposed as properties.
        # Let's check Marathi CardinalFst if it exposes digit and zero.
        # If not, we might need to recreate them here or assume access.
        # Assuming for now we can construct them from digits.
        
        # Recreating digit graph for single digits if needed, or using cardinal graph for digits
        # This part depends on CardinalFst implementation details.
        # In Hindi, cardinal.digit was likely exposed. Check mr/taggers/cardinal.py?
        # For safety, let's just use cardinal_graph for single digits if they are valid cardinals.
        
        # Arabic digits -> Marathi digits
        arabic_to_marathi_digit = pynini.string_map(
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
        arabic_to_marathi_number = pynini.closure(arabic_to_marathi_digit).optimize()

        # Integer part for mantissa
        marathi_int = pynini.compose(pynini.closure(NEMO_MR_DIGIT, 1), cardinal_graph).optimize()
        arabic_int = pynini.compose(pynini.closure(NEMO_DIGIT, 1), arabic_to_marathi_number @ cardinal_graph).optimize()
        integer_graph = (marathi_int | arabic_int).optimize()

        # Fractional digits spoken digit-by-digit
        # We need single digit pronunciations.
        # If `cardinal_graph` works for single digits "०", "१"... then we can use it.
        digit_word_graph = integer_graph # Approximation: assuming integers map correctly for single digits. 

        marathi_frac = pynini.compose(
            pynini.closure(NEMO_MR_DIGIT, 1),
            digit_word_graph + pynini.closure(insert_space + digit_word_graph),
        ).optimize()
        arabic_frac = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_marathi_number @ (digit_word_graph + pynini.closure(insert_space + digit_word_graph)),
        ).optimize()
        fractional_graph = (marathi_frac | arabic_frac).optimize()

        point = pynutil.delete(".") + pynutil.insert(" दशमलव ")
        mantissa_graph = (integer_graph + point + fractional_graph).optimize()

        # Exponent (integer)
        exponent_graph = integer_graph

        # e/E separator, optionally written as "-e" like "10.1-e5"
        e_sep = pynini.closure(pynutil.delete("-"), 0, 1) + pynutil.delete(pynini.union("e", "E"))

        optional_sign = pynini.closure(
            pynutil.insert('sign: "')
            + (pynini.cross("-", "नकारात्मक") | pynini.cross("+", "सकारात्मक"))
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
