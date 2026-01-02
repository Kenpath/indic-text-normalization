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

from indic_text_normalization.kn.graph_utils import (
    NEMO_DIGIT,
    NEMO_KN_DIGIT,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from indic_text_normalization.kn.utils import get_abs_path

# Convert Arabic digits (0-9) to Kannada digits (೦-೯)
arabic_to_kannada_digit = pynini.string_map([
    ("0", "೦"), ("1", "೧"), ("2", "೨"), ("3", "೩"), ("4", "೪"),
    ("5", "೫"), ("6", "೬"), ("7", "೭"), ("8", "೮"), ("9", "೯")
]).optimize()
arabic_to_kannada_number = pynini.closure(arabic_to_kannada_digit).optimize()

# Load math operations
math_operations = pynini.string_file(get_abs_path("data/math_operations.tsv"))


class MathFst(GraphFst):
    """
    Finite state transducer for classifying math expressions, e.g.
        "1=2" -> math { left: "ಒಂದು" operator: "ಸಮಾನ" right: "ಎರಡು" }
        "1+2" -> math { left: "ಒಂದು" operator: "ಪ್ಲಸ್" right: "ಎರಡು" }
        "೧೨=೩೪" -> math { left: "ಹನ್ನೆರಡು" operator: "ಸಮಾನ" right: "ಮೂವತ್ತುನಾಲ್ಕು" }

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="math", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph
        digit_word_graph = (cardinal.digit | cardinal.zero).optimize()
        
        # Support both Kannada and Arabic digits
        # Kannada digits input
        kannada_number_input = pynini.closure(NEMO_KN_DIGIT, 1)
        kannada_number_graph = pynini.compose(kannada_number_input, cardinal_graph).optimize()
        
        # Arabic digits input
        arabic_number_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_number_graph = pynini.compose(
            arabic_number_input,
            arabic_to_kannada_number @ cardinal_graph
        ).optimize()
        
        # Combined integer number graph
        integer_graph = kannada_number_graph | arabic_number_graph

        # Decimal operands: allow x.y where x/y are digit sequences (Kannada or Arabic)
        # Fractional part is spoken digit-by-digit (matches existing decimal verbalizer style).
        kannada_fractional_input = pynini.closure(NEMO_KN_DIGIT, 1)
        kannada_fractional_graph = pynini.compose(
            kannada_fractional_input,
            digit_word_graph + pynini.closure(insert_space + digit_word_graph),
        ).optimize()
        arabic_fractional_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_fractional_graph = pynini.compose(
            arabic_fractional_input,
            arabic_to_kannada_number @ (digit_word_graph + pynini.closure(insert_space + digit_word_graph)),
        ).optimize()
        fractional_graph = (kannada_fractional_graph | arabic_fractional_graph).optimize()

        point = pynutil.delete(".") + pynutil.insert(" ದಶಮಲವ ")
        decimal_graph = (integer_graph + point + fractional_graph).optimize()

        # Operands supported by math expressions
        # Prefer decimals when they match, otherwise fall back to integers.
        operand_graph = (pynutil.add_weight(decimal_graph, -0.1) | integer_graph).optimize()

        # Optional space around operators
        optional_space = pynini.closure(NEMO_SPACE, 0, 1)
        delimiter = optional_space | pynutil.insert(" ")
        tight = pynutil.insert("")  # no space

        # Operators that can appear between numbers
        # Exclude : and / to avoid conflicts with time and dates
        operators = pynini.union("+", "-", "*", "=", "&", "^", "%", "$", "#", "@", "!", "<", ">", "(", ")")
        
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

        # Also support: number operator number operator number (for longer expressions)
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

        # Special-case: tight dash patterns
        # Pattern 1: "10-2=8" should be treated as "ರಿಂದ" (from) - tight minus with equals
        extended_math_tight_range = (
            pynutil.insert("left: \"")
            + operand_graph
            + pynutil.insert("\"")
            + tight
            + pynutil.insert("operator: \"")
            + pynini.cross("-", "ರಿಂದ")
            + pynutil.insert("\"")
            + tight
            + pynutil.insert("middle: \"")
            + operand_graph
            + pynutil.insert("\"")
            + tight
            + pynutil.insert("operator_two: \"")
            + pynini.cross("=", "ಸಮಾನ")
            + pynutil.insert("\"")
            + tight
            + pynutil.insert("right: \"")
            + operand_graph
            + pynutil.insert("\"")
        )

        # Pattern 2: "10-2 ದೊಡ್ಡ ಸಂಖ್ಯೆ" should also be treated as "ರಿಂದ" (from) - tight minus without equals
        # This matches number-number (no spaces around "-") and outputs a math token for just the pair.
        math_expression_tight_minus_text = (
            pynutil.insert("left: \"")
            + operand_graph
            + pynutil.insert("\"")
            + tight
            + pynutil.insert("operator: \"")
            + pynini.cross("-", "ರಿಂದ")
            + pynutil.insert("\"")
            + tight
            + pynutil.insert("right: \"")
            + operand_graph
            + pynutil.insert("\"")
        )

        final_graph = (
            pynutil.add_weight(extended_math_tight_range, -0.2)
            | pynutil.add_weight(math_expression_tight_minus_text, -0.15)
            | math_expression
            | extended_math
        )
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

