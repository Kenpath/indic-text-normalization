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

from ..graph_utils import (
    NEMO_DIGIT,
    NEMO_BRX_DIGIT,
    NEMO_HI_DIGIT,
    NEMO_SPACE,
    NEMO_SIGMA,
    NEMO_SUPERSCRIPT_DIGIT,
    NEMO_SUPERSCRIPT_MINUS,
    NEMO_SUPERSCRIPT_PLUS,
    superscript_to_digit,
    superscript_to_sign,
    GraphFst,
    insert_space,
    delete_zero_or_one_space,
)
from ..utils import get_abs_path

# Convert Arabic digits (0-9) to Bodo digits (०-९)
arabic_to_brx_digit = pynini.string_map([
    ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
    ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९")
]).optimize()
arabic_to_brx_number = pynini.closure(arabic_to_brx_digit).optimize()

# Keep old names for backward compatibility
arabic_to_hindi_number = arabic_to_brx_number

class MathFst(GraphFst):
    """
    Finite state transducer for classifying math expressions.
    Ensures every field value is followed by a space to satisfy TokenParser.

    Args:
        cardinal: cardinal GraphFst
        decimal: decimal GraphFst (optional)
        deterministic: if True will provide a single transduction option
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst = None, deterministic: bool = True):
        super().__init__(name="math", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph
        digit_word_graph = (cardinal.digit | cardinal.zero).optimize()
        
        # Support both native and Arabic digits
        # Native digits input
        native_number_input = pynini.closure(NEMO_HI_DIGIT, 1)
        native_number_graph = pynini.compose(native_number_input, cardinal_graph).optimize()
        
        # Arabic digits input
        arabic_number_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_number_graph = pynini.compose(
            arabic_number_input,
            arabic_to_brx_number @ cardinal_graph
        ).optimize()
        
        # Combined integer graph
        integer_graph = (native_number_graph | arabic_number_graph).optimize()

        # Decimal support inside math (needed for π equations)
        # Speak fractional digits digit-by-digit and use "दशमलव" as decimal separator.
        native_frac = pynini.compose(
            pynini.closure(NEMO_HI_DIGIT, 1),
            digit_word_graph + pynini.closure(insert_space + digit_word_graph),
        ).optimize()
        arabic_frac = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_brx_number @ (digit_word_graph + pynini.closure(insert_space + digit_word_graph)),
        ).optimize()
        fractional_graph = (native_frac | arabic_frac).optimize()

        point = pynutil.delete(".") + pynutil.insert(" दशमलव ")
        decimal_graph = (integer_graph + point + fractional_graph).optimize()

        # Number operands: prefer decimals when they match, otherwise fall back to integers.
        # Give decimals higher weight (lower number = higher priority) to ensure they match before scientific notation
        number_graph = (pynutil.add_weight(decimal_graph, -0.2) | integer_graph).optimize()

        # Load Greek letters to allow them as operands (e.g., π = 3.14)
        # We need both the input (to match) and output (translation) for math expressions
        greek_letters_translation = pynini.string_file(get_abs_path("data/greek.tsv")).optimize()
        # For operands, we want the translation (e.g., π -> "पाइ")
        # Left operand can be Greek letter (translated) or number
        # Right operand should be number only (to avoid "π = λ" matching when we want "π = 3.14")
        left_operand_graph = (number_graph | greek_letters_translation).optimize()
        right_operand_graph = number_graph.optimize()
        
        # General operand graph (for cases where either side can be number or Greek)
        operand_graph = left_operand_graph.optimize()

        math_operations = pynini.string_file(get_abs_path("data/math_operations.tsv"))

        # Optional space around operators
        optional_space = pynini.closure(NEMO_SPACE, 0, 1)
        delimiter = optional_space | pynutil.insert(" ")
        
        # Operators that can appear between numbers
        operators = pynini.union("+", "-", "*", "÷", "×", "=", "&", "^", "%", "$", "#", "@", "!", "<", ">", ",", "(", ")", "?", "≈", "√", "·")
        operators = operators.optimize()

        # Support for power expressions (e.g., 10⁻⁷, 2³)
        superscript_sign = pynini.closure(superscript_to_sign, 0, 1)
        superscript_number = pynini.closure(superscript_to_digit, 1)
        
        power_expression = (
            pynutil.insert("left: \"")
            + number_graph
            + pynutil.insert("\" ")
            + pynutil.insert("operator: \"पावर\" ")
            + pynutil.insert("right: \"")
            + (superscript_sign @ pynini.cdrewrite(pynini.cross("-", "ऋणात्मक "), "", "", NEMO_SIGMA))
            + (superscript_number @ cardinal_graph)
            + pynutil.insert("\" ")
        )

        # Math expression: operand operator operand
        # Left operand can be Greek letter or number, right operand should be number
        math_expression = (
            pynutil.insert("left: \"")
            + left_operand_graph
            + pynutil.insert("\" ")
            + delimiter
            + pynutil.insert("operator: \"")
            + (operators @ math_operations)
            + pynutil.insert("\" ")
            + delimiter
            + pynutil.insert("right: \"")
            + right_operand_graph
            + pynutil.insert("\" ")
        )

        # Extended math (e.g., 1+2+3 or π+2+3)
        # Left can be Greek letter or number, middle and right should be numbers
        extended_math = (
            pynutil.insert("left: \"")
            + left_operand_graph
            + pynutil.insert("\" ")
            + delimiter
            + pynutil.insert("operator: \"")
            + (operators @ math_operations)
            + pynutil.insert("\" ")
            + delimiter
            + pynutil.insert("middle: \"")
            + right_operand_graph
            + pynutil.insert("\" ")
            + delimiter
            + pynutil.insert("operator_two: \"")
            + (operators @ math_operations)
            + pynutil.insert("\" ")
            + delimiter
            + pynutil.insert("right: \"")
            + right_operand_graph
            + pynutil.insert("\" ")
        )

        # Support: operator number (e.g., "+5", "√9")
        operator_number = (
            pynutil.insert("left: \"")
            + pynutil.insert("")
            + pynutil.insert("\" ")
            + pynutil.insert("operator: \"")
            + (operators @ math_operations)
            + pynutil.insert("\" ")
            + delimiter
            + pynutil.insert("right: \"")
            + right_operand_graph
            + pynutil.insert("\" ")
        )

        # Support: number operator (e.g., "5+")
        number_operator = (
            pynutil.insert("left: \"")
            + right_operand_graph
            + pynutil.insert("\" ")
            + delimiter
            + pynutil.insert("operator: \"")
            + (operators @ math_operations)
            + pynutil.insert("\" ")
            + pynutil.insert("right: \"")
            + pynutil.insert("")
            + pynutil.insert("\" ")
        )

        # Support: standalone operator
        standalone_operator = (
            pynutil.insert("left: \"")
            + pynutil.insert("")
            + pynutil.insert("\" ")
            + pynutil.insert("operator: \"")
            + (operators @ math_operations)
            + pynutil.insert("\" ")
            + pynutil.insert("right: \"")
            + pynutil.insert("")
            + pynutil.insert("\" ")
        )

        # Root expressions: √2, √3, etc. (square root)
        sqrt_symbol = pynini.accep("√")
        optional_space_after_sqrt = pynini.closure(NEMO_SPACE, 0, 1)
        sqrt_expression = (
            pynutil.insert("left: \"")
            + pynutil.insert("")
            + pynutil.insert("\" ")
            + pynutil.insert("operator: \"")
            + pynini.cross(sqrt_symbol, "वर्गमूल")
            + pynutil.insert("\" ")
            + optional_space_after_sqrt
            + pynutil.insert("right: \"")
            + right_operand_graph
            + pynutil.insert("\" ")
        )

        # Special-case: tight dash patterns
        # Pattern 1: "10-2=8" should be treated as "दानख" (minus) - tight minus with equals
        # These are number-number patterns, so use right_operand_graph
        math_expression_tight_minus_equals = (
            pynutil.insert("left: \"")
            + right_operand_graph
            + pynutil.insert("\" ")
            + pynutil.insert("operator: \"")
            + pynini.cross("-", "दानख")
            + pynutil.insert("\" ")
            + pynutil.insert("middle: \"")
            + right_operand_graph
            + pynutil.insert("\" ")
            + pynutil.insert("operator_two: \"")
            + pynini.cross("=", "समान")
            + pynutil.insert("\" ")
            + pynutil.insert("right: \"")
            + right_operand_graph
            + pynutil.insert("\" ")
        )

        # Pattern 2: "10-2 गेदेर संख्या" should be treated as "से" (from) - tight minus without equals
        # This matches number-number (no spaces around "-") and outputs a math token for just the pair.
        math_expression_tight_minus_text = (
            pynutil.insert("left: \"")
            + right_operand_graph
            + pynutil.insert("\" ")
            + pynutil.insert("operator: \"")
            + pynini.cross("-", "से")
            + pynutil.insert("\" ")
            + pynutil.insert("right: \"")
            + right_operand_graph
            + pynutil.insert("\" ")
        )

        final_graph = (
            pynutil.add_weight(sqrt_expression, -0.1)
            | pynutil.add_weight(math_expression_tight_minus_equals, -0.2)
            | pynutil.add_weight(math_expression_tight_minus_text, -0.15)
            | pynutil.add_weight(power_expression, -0.1)
            | math_expression
            | extended_math
            | operator_number
            | number_operator
            | standalone_operator
        )
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
