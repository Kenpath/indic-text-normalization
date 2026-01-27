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

from indic_text_normalization.gu.graph_utils import (
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_GU_DIGIT,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from indic_text_normalization.gu.utils import get_abs_path

# Convert Arabic digits (0-9) to Gujarati digits (૦-૯)
arabic_to_gujarati_digit = pynini.string_map([
    ("0", "૦"), ("1", "૧"), ("2", "૨"), ("3", "૩"), ("4", "૪"),
    ("5", "૫"), ("6", "૬"), ("7", "૭"), ("8", "૮"), ("9", "૯")
]).optimize()
arabic_to_gujarati_number = pynini.closure(arabic_to_gujarati_digit).optimize()

# Load math operations and Greek letters
math_operations = pynini.string_file(get_abs_path("data/math_operations.tsv"))
greek_letters = pynini.string_file(get_abs_path("data/greek.tsv"))


class MathFst(GraphFst):
    """
    Finite state transducer for classifying math expressions, e.g.
        "1=2" -> math { left: "એક" operator: "બરાબર" right: "બે" }
        "1+2" -> math { left: "એક" operator: "વત્તા" right: "બે" }
        "૧૨=૩૪" -> math { left: "બાર" operator: "બરાબર" right: "ચોત્રીસ" }

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="math", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph
        
        # Support both Gujarati and Arabic digits
        # Gujarati digits input
        gujarati_number_input = pynini.closure(NEMO_GU_DIGIT, 1)
        gujarati_number_graph = pynini.compose(gujarati_number_input, cardinal_graph).optimize()
        
        # Arabic digits input
        arabic_number_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_number_graph = pynini.compose(
            arabic_number_input,
            arabic_to_gujarati_number @ cardinal_graph
        ).optimize()
        
        # Combined number graph
        number_graph = gujarati_number_graph | arabic_number_graph

        # Decimal support inside math (needed for π equations)
        # Speak fractional digits digit-by-digit and use "દશાંશ" as decimal separator.
        cardinal_digit_graph = (cardinal.digit | cardinal.zero).optimize()
        
        gujarati_frac = pynini.compose(
            pynini.closure(NEMO_GU_DIGIT, 1),
            cardinal_digit_graph + pynini.closure(insert_space + cardinal_digit_graph),
        ).optimize()
        
        arabic_frac = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_gujarati_number @ (cardinal_digit_graph + pynini.closure(insert_space + cardinal_digit_graph)),
        ).optimize()
        
        fractional_graph = (gujarati_frac | arabic_frac).optimize()

        point = pynutil.delete(".") + pynutil.insert(" દશાંશ ")
        decimal_graph = (number_graph + point + fractional_graph).optimize()

        # Greek letters support
        greek_graph = greek_letters

        # Square root support moved to operators

        # Operators that can appear between numbers
        # Added: × (times), ÷ (divide), √ (sqrt), ≈ (approx), · (dot product), x/X (multiplication)
        operators = pynini.union("+", "-", "*", "=", "&", "^", "%", "$", "#", "@", "!", "<", ">", ",", "(", ")", "?", "×", "÷", "√", "≈", "·", "x", "X")

        # Extract just the Greek characters (input side) from the mapping
        greek_char = pynini.project(greek_letters, "input")
        
        # Alpha char should NOT exclude 'x' or 'X' even though they are operators now
        # because we want to support 'x' as a variable too (e.g. sqrt(x))
        operators_excluding_x = pynini.difference(operators, pynini.union("x", "X"))

        alpha_char = pynini.difference(
            NEMO_CHAR, 
            operators_excluding_x | NEMO_SPACE | NEMO_DIGIT | NEMO_GU_DIGIT | greek_char
        ).optimize()
        alpha_graph = pynini.closure(alpha_char, 1)

        # Operands supported by math expressions
        # Prefer decimals when they match (weight -0.1), otherwise fall back to other types
        operand_graph = pynutil.add_weight(decimal_graph, -0.1) | number_graph | greek_graph | alpha_graph

        # Optional space around operators
        optional_space = pynini.closure(NEMO_SPACE, 0, 1)
        delimiter = optional_space | pynutil.insert(" ")
        
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
        # This handles cases like "1+2+3" or "π=3.14"
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

        # Support: operator number (e.g., "+5", "*3", "√x")
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

        # Special-case: tight dash patterns (similar to Hindi)
        # Use simpler number_graph for tight patterns to avoid complexity issues
        tight = pynutil.insert("")  # no space
        
        # Pattern 1: "10-2=8" - tight minus with equals (no spaces)
        # Use number_graph (not operand_graph) for simpler matching
        math_expression_tight_minus_equals = (
            pynutil.insert("left: \"")
            + number_graph
            + pynutil.insert("\" ")
            + pynutil.insert("operator: \"")
            + pynini.cross("-", "થી")
            + pynutil.insert("\" ")
            + pynutil.insert("middle: \"")
            + number_graph
            + pynutil.insert("\" ")
            + pynutil.insert("operator_two: \"")
            + pynini.cross("=", "બરાબર")
            + pynutil.insert("\" ")
            + pynutil.insert("right: \"")
            + number_graph
            + pynutil.insert("\" ")
        )

        # Pattern 2: "10-2 ગેદેર સંખ્યા" - tight minus followed by text
        math_expression_tight_minus_text = (
            pynutil.insert("left: \"")
            + number_graph
            + pynutil.insert("\" ")
            + pynutil.insert("operator: \"")
            + pynini.cross("-", "થી")
            + pynutil.insert("\" ")
            + pynutil.insert("right: \"")
            + number_graph
            + pynutil.insert("\" ")
        )

        # Pattern 3: "10 - 7 = 3" - spaced minus with equals
        spaced_math_minus_equals = (
            pynutil.insert("left: \"")
            + number_graph
            + pynutil.insert("\" ")
            + pynutil.delete(" ")
            + pynutil.insert("operator: \"")
            + pynini.cross("-", "બાદબાકી")
            + pynutil.insert("\" ")
            + pynutil.delete(" ")
            + pynutil.insert("middle: \"")
            + number_graph
            + pynutil.insert("\" ")
            + pynutil.delete(" ")
            + pynutil.insert("operator_two: \"")
            + pynini.cross("=", "બરાબર")
            + pynutil.insert("\" ")
            + pynutil.delete(" ")
            + pynutil.insert("right: \"")
            + number_graph
            + pynutil.insert("\" ")
        )

        # Square root expressions: √2, √3, etc.
        sqrt_symbol = pynini.accep("√")
        optional_space_after_sqrt = pynini.closure(NEMO_SPACE, 0, 1)
        sqrt_expression = (
            pynutil.insert("left: \"")
            + pynutil.insert("")
            + pynutil.insert("\" ")
            + pynutil.insert("operator: \"")
            + pynini.cross(sqrt_symbol, "વર્ગમૂળ")
            + pynutil.insert("\" ")
            + optional_space_after_sqrt
            + pynutil.insert("right: \"")
            + number_graph
            + pynutil.insert("\" ")
        )

        final_graph = (
            pynutil.add_weight(sqrt_expression, -0.25)
            | pynutil.add_weight(math_expression_tight_minus_equals, -0.2)
            | pynutil.add_weight(spaced_math_minus_equals, -0.18)
            | pynutil.add_weight(math_expression_tight_minus_text, -0.15)
            | math_expression
            | extended_math
            | operator_number
            | number_operator
            | standalone_operator
        )
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
