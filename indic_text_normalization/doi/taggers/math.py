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

from indic_text_normalization.doi.graph_utils import (
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_HI_DIGIT,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from indic_text_normalization.doi.utils import get_abs_path

# Convert Arabic digits (0-9) to Dogri digits (०-९)
arabic_to_dogri_digit = pynini.string_map([
    ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
    ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९")
]).optimize()
arabic_to_dogri_number = pynini.closure(arabic_to_dogri_digit).optimize()

# Load math operations and Greek letters
math_operations = pynini.string_file(get_abs_path("data/math_operations.tsv"))
greek_letters = pynini.string_file(get_abs_path("data/greek.tsv"))


class MathFst(GraphFst):
    """
    Finite state transducer for classifying math expressions, e.g.
        "1=2" -> math { left: "एक" operator: "बराबर" right: "दो" }
        "1+2" -> math { left: "एक" operator: "प्लस" right: "दो" }
        "१२=३४" -> math { left: "बारह" operator: "बराबर" right: "चौंतीस" }

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="math", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph
        
        # Support both Dogri and Arabic digits
        # Dogri digits input
        dogri_number_input = pynini.closure(NEMO_HI_DIGIT, 1)
        dogri_number_graph = pynini.compose(dogri_number_input, cardinal_graph).optimize()
        
        # Arabic digits input
        arabic_number_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_number_graph = pynini.compose(
            arabic_number_input,
            arabic_to_dogri_number @ cardinal_graph
        ).optimize()
        
        # Combined number graph
        number_graph = dogri_number_graph | arabic_number_graph

        # Decimal support inside math (needed for π equations)
        # Speak fractional digits digit-by-digit and use "दशमलव" as decimal separator.
        cardinal_digit_graph = (cardinal.digit | cardinal.zero).optimize()
        
        dogri_frac = pynini.compose(
            pynini.closure(NEMO_HI_DIGIT, 1),
            cardinal_digit_graph + pynini.closure(insert_space + cardinal_digit_graph),
        ).optimize()
        
        arabic_frac = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_dogri_number @ (cardinal_digit_graph + pynini.closure(insert_space + cardinal_digit_graph)),
        ).optimize()
        
        fractional_graph = (dogri_frac | arabic_frac).optimize()

        point = pynutil.delete(".") + pynutil.insert(" दशमलव ")
        decimal_graph = (number_graph + point + fractional_graph).optimize()

        # Greek letters support
        greek_graph = greek_letters

        # Greek letters support
        greek_graph = greek_letters

        # Square root support moved to operators


        # Operands supported by math expressions: numbers, Greek letters, square root, or alphanumeric strings
        # Alphanumeric support for mixed scripts/words in math (e.g. "pi = 3.14... text")
        # Exclude operators, space, and digits to avoid ambiguity
        
        # Operators that can appear between numbers (moved up for alpha definition)
        # Operators that can appear between numbers (moved up for alpha definition)
        # Added: × (times), ÷ (divide), √ (sqrt), ≈ (approx), · (dot product), x/X (multiplication)
        operators = pynini.union("+", "-", "*", "=", "&", "^", "%", "$", "#", "@", "!", "<", ">", ",", "(", ")", "?", "×", "÷", "√", "≈", "·", "x", "X")

        # Extract just the Greek characters (input side) from the mapping
        greek_char = pynini.project(greek_letters, "input")
        
        # Alpha char should NOT exclude 'x' or 'X' even though they are operators now
        # because we want to support 'x' as a variable too (e.g. sqrt(x))
        operators_excluding_x = pynini.difference(operators, pynini.union("x", "X"))

        alpha_char = pynini.difference(
            NEMO_CHAR, 
            operators_excluding_x | NEMO_SPACE | NEMO_DIGIT | NEMO_HI_DIGIT | greek_char
        ).optimize()
        alpha_graph = pynini.closure(alpha_char, 1)

        # Operands supported by math expressions
        # Prefer decimals when they match (weight -0.1), otherwise fall back to other types
        # Operands supported by math expressions
        # Prefer decimals when they match (weight -0.1), otherwise fall back to other types
        # Removed sqrt_graph (now an operator)
        operand_graph = pynutil.add_weight(decimal_graph, -0.1) | number_graph | greek_graph | alpha_graph

        # Optional space around operators
        optional_space = pynini.closure(NEMO_SPACE, 0, 1)
        delimiter = optional_space | pynutil.insert(" ")

        # Operators that can appear between numbers
        # Exclude : and / to avoid conflicts with time and dates
        # operators definition moved up

        
        # Math expression: number operator number
        # Pattern: number [space] operator [space] number
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

        # Support: operator number (e.g., "+5", "*3")
        operator_number = (
            pynutil.insert("left: \"")
            + pynutil.insert("")
            + pynutil.insert("\"")
            + pynutil.insert("operator: \"")
            + (operators @ math_operations)
            + pynutil.insert("\"")
            + delimiter
            + pynutil.insert("right: \"")
            + operand_graph
            + pynutil.insert("\"")
        )

        # Support: number operator (e.g., "5+", "3*")
        number_operator = (
            pynutil.insert("left: \"")
            + operand_graph
            + pynutil.insert("\"")
            + delimiter
            + pynutil.insert("operator: \"")
            + (operators @ math_operations)
            + pynutil.insert("\"")
            + pynutil.insert("right: \"")
            + pynutil.insert("")
            + pynutil.insert("\"")
        )

        # Support: standalone operator (e.g., "+", "*", "?")
        standalone_operator = (
            pynutil.insert("left: \"")
            + pynutil.insert("")
            + pynutil.insert("\"")
            + pynutil.insert("operator: \"")
            + (operators @ math_operations)
            + pynutil.insert("\"")
            + pynutil.insert("right: \"")
            + pynutil.insert("")
            + pynutil.insert("\"")
        )

        # Special-case: tight dash patterns (from Hindi implementation)
        tight = pynutil.insert("")  # no space
        # Pattern 1: "10-2=8" should be treated as "से" (from) - tight minus with equals
        math_expression_tight_minus_equals = (
            pynutil.insert("left: \"")
            + operand_graph
            + pynutil.insert("\"")
            + tight
            + pynutil.insert("operator: \"")
            + pynini.cross("-", "से")
            + pynutil.insert("\"")
            + tight
            + pynutil.insert("middle: \"")
            + operand_graph
            + pynutil.insert("\"")
            + tight
            + pynutil.insert("operator_two: \"")
            + pynini.cross("=", "बराबर")
            + pynutil.insert("\"")
            + tight
            + pynutil.insert("right: \"")
            + operand_graph
            + pynutil.insert("\"")
        )

        # Pattern 2: "10-2 गेदेर संख्या" should also be treated as "से" (from) - tight minus without equals
        # This matches operand-operand (no spaces around "-") and outputs a math token for just the pair.
        math_expression_tight_minus_text = (
            pynutil.insert("left: \"")
            + operand_graph
            + pynutil.insert("\"")
            + tight
            + pynutil.insert("operator: \"")
            + pynini.cross("-", "से")
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

