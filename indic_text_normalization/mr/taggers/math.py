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

from indic_text_normalization.mr.graph_utils import (
    NEMO_DIGIT,
    NEMO_MR_DIGIT,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from indic_text_normalization.mr.utils import get_abs_path

# Convert Arabic digits (0-9) to Marathi digits (०-९)
arabic_to_marathi_digit = pynini.string_map([
    ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
    ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९")
]).optimize()
arabic_to_marathi_number = pynini.closure(arabic_to_marathi_digit).optimize()

# Load math operations
math_operations = pynini.string_file(get_abs_path("data/math_operations.tsv"))


class MathFst(GraphFst):
    """
    Finite state transducer for classifying math expressions, e.g.
        "1=2" -> math { left: "एक" operator: "बराबर" right: "दोन" }
        "1+2" -> math { left: "एक" operator: "अधिक" right: "दोन" }
        "१२=३४" -> math { left: "बारा" operator: "बराबर" right: "चौतीस" }

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="math", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph
        
        # Support both Marathi and Arabic digits
        # Marathi digits input
        marathi_number_input = pynini.closure(NEMO_MR_DIGIT, 1)
        marathi_number_graph = pynini.compose(marathi_number_input, cardinal_graph).optimize()
        
        # Arabic digits input
        arabic_number_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_number_graph = pynini.compose(
            arabic_number_input,
            arabic_to_marathi_number @ cardinal_graph
        ).optimize()
        
        # Combined number graph
        number_graph = marathi_number_graph | arabic_number_graph

        # Operands supported by math expressions
        operand_graph = number_graph

        # Optional space around operators
        optional_space = pynini.closure(NEMO_SPACE, 0, 1)
        delimiter = optional_space | pynutil.insert(" ")

        # Operators that can appear between numbers
        # Exclude : and / to avoid conflicts with time and dates
        operators = pynini.union("+", "-", "*", "×", "÷", "=", "&", "^", "%", "$", "#", "@", "!", "<", ">", ",", "(", ")", "?")
        
        # Math expression: operand operator operand
        # Pattern: operand [space] operator [space] operand
        math_expression = (
            pynutil.insert("left: \"")
            + operand_graph
            + pynutil.insert("\"")
            + delimiter
            + pynutil.insert("operator: \"")
            + (operators @ math_operations)
            + pynutil.insert("\"")
            + delimiter
            + pynutil.insert("right: \"")
            + operand_graph
            + pynutil.insert("\"")
        )

        # Also support: operand operator operand operator operand (for longer expressions)
        # This handles cases like "1+2+3"
        extended_math = (
            pynutil.insert("left: \"")
            + operand_graph
            + pynutil.insert("\"")
            + delimiter
            + pynutil.insert("operator: \"")
            + (operators @ math_operations)
            + pynutil.insert("\"")
            + delimiter
            + pynutil.insert("middle: \"")
            + operand_graph
            + pynutil.insert("\"")
            + delimiter
            + pynutil.insert("operator_two: \"")
            + (operators @ math_operations)
            + pynutil.insert("\"")
            + delimiter
            + pynutil.insert("right: \"")
            + operand_graph
            + pynutil.insert("\"")
        )

        # Support: operator number (e.g., "+5", "*3")
        operator_number = (
            pynutil.insert("left: \"")
            + pynutil.insert("")
            + pynutil.insert("\"")
            + insert_space
            + pynutil.insert("operator: \"")
            + (operators @ math_operations)
            + pynutil.insert("\"")
            + delimiter
            + pynutil.insert("right: \"")
            + number_graph
            + pynutil.insert("\"")
        )

        # Support: number operator (e.g., "5+", "3*")
        number_operator = (
            pynutil.insert("left: \"")
            + number_graph
            + pynutil.insert("\"")
            + delimiter
            + pynutil.insert("operator: \"")
            + (operators @ math_operations)
            + pynutil.insert("\"")
            + insert_space
            + pynutil.insert("right: \"")
            + pynutil.insert("")
            + pynutil.insert("\"")
        )

        # Support: standalone operator (e.g., "+", "*", "?")
        standalone_operator = (
            pynutil.insert("left: \"")
            + pynutil.insert("")
            + pynutil.insert("\"")
            + insert_space
            + pynutil.insert("operator: \"")
            + (operators @ math_operations)
            + pynutil.insert("\"")
            + insert_space
            + pynutil.insert("right: \"")
            + pynutil.insert("")
            + pynutil.insert("\"")
        )

        # Special-case: tight dash patterns
        tight = insert_space  # space for parser safety
        # Pattern 1: "10-2=8" should be treated as "पासून" (from) - tight minus with equals
        math_expression_tight_minus_equals = (
            pynutil.insert("left: \"")
            + operand_graph
            + pynutil.insert("\"")
            + tight
            + pynutil.insert("operator: \"")
            + pynini.cross("-", "पासून")
            + pynutil.insert("\"")
            + tight
            + pynutil.insert("middle: \"")
            + operand_graph
            + pynutil.insert("\"")
            + tight
            + pynutil.insert("operator_two: \"")
            + pynini.cross("=", "बरोबर")
            + pynutil.insert("\"")
            + tight
            + pynutil.insert("right: \"")
            + operand_graph
            + pynutil.insert("\"")
        )

        # Pattern 2: "10-2 बड़ी संख्या" should also be treated as "से" -> "पासून" (from) - tight minus without equals
        # This matches number-number (no spaces around "-") and outputs a math token for just the pair.
        math_expression_tight_minus_text = (
            pynutil.insert("left: \"")
            + operand_graph
            + pynutil.insert("\"")
            + tight
            + pynutil.insert("operator: \"")
            + pynini.cross("-", "पासून")
            + pynutil.insert("\"")
            + tight
            + pynutil.insert("right: \"")
            + operand_graph
            + pynutil.insert("\"")
        )

        final_graph = (
            pynutil.add_weight(math_expression_tight_minus_equals, -0.2)
            | pynutil.add_weight(math_expression_tight_minus_text, -0.15)
            | math_expression
            | extended_math
            | operator_number
            | number_operator
            | standalone_operator
        )
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
