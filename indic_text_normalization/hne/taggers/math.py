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

from indic_text_normalization.hne.graph_utils import (
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_CG_DIGIT,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from indic_text_normalization.hne.utils import get_abs_path

# Convert Arabic digits (0-9) to Chhattisgarhi digits (०-९)
arabic_to_cg_digit = pynini.string_map([
    ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
    ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९")
]).optimize()
arabic_to_cg_number = pynini.closure(arabic_to_cg_digit).optimize()

# Load math operations and Greek letters
math_operations = pynini.string_file(get_abs_path("data/math_operations.tsv"))
greek_letters = pynini.string_file(get_abs_path("data/greek.tsv"))


class MathFst(GraphFst):
    """
    Finite state transducer for classifying math expressions, e.g.
        "1=2" -> math { left: "एक" operator: "बराबर" right: "दुई" }
        "1+2" -> math { left: "एक" operator: "जोड़" right: "दुई" }
        "१२=३४" -> math { left: "बारह" operator: "बराबर" right: "चौंतीस" }
        "λ = 5" -> math { left: "लैम्ब्डा" operator: "बराबर" right: "पाँच" }

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="math", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph
        
        # Support both Chhattisgarhi and Arabic digits
        # Chhattisgarhi digits input
        chhattisgarhi_number_input = pynini.closure(NEMO_CG_DIGIT, 1)
        chhattisgarhi_number_graph = pynini.compose(chhattisgarhi_number_input, cardinal_graph).optimize()
        
        # Arabic digits input
        arabic_number_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_number_graph = pynini.compose(
            arabic_number_input,
            arabic_to_cg_number @ cardinal_graph
        ).optimize()
        
        # Combined number graph
        number_graph = chhattisgarhi_number_graph | arabic_number_graph

        # Decimal support inside math (needed for π equations)
        # Speak fractional digits digit-by-digit and use "दशमलव" as decimal separator.
        cardinal_digit_graph = (cardinal.digit | cardinal.zero).optimize()
        
        cg_frac = pynini.compose(
            pynini.closure(NEMO_CG_DIGIT, 1),
            cardinal_digit_graph + pynini.closure(insert_space + cardinal_digit_graph),
        ).optimize()
        
        arabic_frac = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_cg_number @ (cardinal_digit_graph + pynini.closure(insert_space + cardinal_digit_graph)),
        ).optimize()
        
        fractional_graph = (cg_frac | arabic_frac).optimize()

        point = pynutil.delete(".") + pynutil.insert(" दशमलव ")
        decimal_graph = (number_graph + point + fractional_graph).optimize()

        # Greek letters support
        greek_graph = greek_letters

        # Operators that can appear between numbers
        # Added: × (times), ÷ (divide), √ (sqrt), ≈ (approx), · (dot product), x/X (multiplication)
        # Note: commas are handled as punctuation separators to allow long lists.
        operators = pynini.union("+", "-", "*", "=", "&", "^", "%", "$", "#", "@", "!", "<", ">", "(", ")", "?", "×", "÷", "√", "≈", "·", "x", "X")

        # Extract just the Greek characters (input side) from the mapping
        greek_char = pynini.project(greek_letters, "input")
        
        # Alpha char should NOT exclude 'x' or 'X' even though they are operators now
        # because we want to support 'x' as a variable too (e.g. sqrt(x))
        operators_excluding_x = pynini.difference(operators, pynini.union("x", "X"))

        alpha_char = pynini.difference(
            NEMO_CHAR, 
            operators_excluding_x | NEMO_SPACE | NEMO_DIGIT | NEMO_CG_DIGIT | greek_char
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
            + pynini.cross("-", "से")
            + pynutil.insert("\" ")
            + pynutil.insert("middle: \"")
            + number_graph
            + pynutil.insert("\" ")
            + pynutil.insert("operator_two: \"")
            + pynini.cross("=", "बराबर")
            + pynutil.insert("\" ")
            + pynutil.insert("right: \"")
            + number_graph
            + pynutil.insert("\" ")
        )

        # Pattern 2: "10-2 बड़ी संख्या" - tight minus followed by text
        math_expression_tight_minus_text = (
            pynutil.insert("left: \"")
            + number_graph
            + pynutil.insert("\" ")
            + pynutil.insert("operator: \"")
            + pynini.cross("-", "से")
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
            + pynini.cross("-", "घटाव")
            + pynutil.insert("\" ")
            + pynutil.delete(" ")
            + pynutil.insert("middle: \"")
            + number_graph
            + pynutil.insert("\" ")
            + pynutil.delete(" ")
            + pynutil.insert("operator_two: \"")
            + pynini.cross("=", "बराबर")
            + pynutil.insert("\" ")
            + pynutil.delete(" ")
            + pynutil.insert("right: \"")
            + number_graph
            + pynutil.insert("\" ")
        )

        # Square root expressions: √2, √3, √x, √ x, etc.
        sqrt_symbol = pynini.accep("√")
        # Accept optional space after sqrt (keeps it in output, verbalizer handles spacing)
        optional_space_sqrt = pynini.closure(pynutil.delete(NEMO_SPACE), 0, 1)
        
        # Simple variable (single letter a-z, A-Z, or x, y etc)
        single_var = pynini.union(*"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        
        # Combined sqrt operand: number, single variable, or Greek letter
        sqrt_operand = number_graph | single_var | greek_graph
        
        # Basic sqrt expression: √2, √ 2, √x, √ x, √π, √ λ
        sqrt_expression = (
            pynutil.insert("left: \"")
            + pynutil.insert("")
            + pynutil.insert("\" ")
            + pynutil.insert("operator: \"")
            + pynini.cross(sqrt_symbol, "वर्गमूळ")
            + pynutil.insert("\" ")
            + optional_space_sqrt
            + pynutil.insert("right: \"")
            + sqrt_operand
            + pynutil.insert("\" ")
        )
        
        # Sqrt followed by spaced operator: √ x - 3, √2 - 2 (spaces around operator)
        sqrt_with_spaced_operation = (
            pynutil.insert("left: \"")
            + pynini.cross(sqrt_symbol, "वर्गमूळ ")
            + optional_space_sqrt
            + sqrt_operand
            + pynutil.insert("\" ")
            + pynutil.delete(" ")
            + pynutil.insert("operator: \"")
            + (operators @ math_operations)
            + pynutil.insert("\" ")
            + pynutil.delete(" ")
            + pynutil.insert("right: \"")
            + sqrt_operand
            + pynutil.insert("\" ")
        )
        
        # Sqrt followed by tight operator: √2-2 (no spaces at all)
        sqrt_with_tight_operation = (
            pynutil.insert("left: \"")
            + pynini.cross(sqrt_symbol, "वर्गमूळ ")
            + sqrt_operand
            + pynutil.insert("\" ")
            + pynutil.insert("operator: \"")
            + (operators @ math_operations)
            + pynutil.insert("\" ")
            + pynutil.insert("right: \"")
            + sqrt_operand
            + pynutil.insert("\" ")
        )

        # Implicit multiplication: 2x -> "दुई गुना x", 3y -> "तीन गुना y"
        implicit_mult = (
            pynutil.insert("left: \"")
            + number_graph
            + pynutil.insert("\" ")
            + pynutil.insert("operator: \"")
            + pynutil.insert("गुना")
            + pynutil.insert("\" ")
            + pynutil.insert("right: \"")
            + single_var
            + pynutil.insert("\" ")
        )
        
        # Implicit multiplication with Greek: 2π -> "दुई गुना पाई"
        implicit_mult_greek = (
            pynutil.insert("left: \"")
            + number_graph
            + pynutil.insert("\" ")
            + pynutil.insert("operator: \"")
            + pynutil.insert("गुना")
            + pynutil.insert("\" ")
            + pynutil.insert("right: \"")
            + greek_graph
            + pynutil.insert("\" ")
        )

        final_graph = (
            pynutil.add_weight(sqrt_with_spaced_operation, -0.28)
            | pynutil.add_weight(sqrt_with_tight_operation, -0.27)
            | pynutil.add_weight(sqrt_expression, -0.25)
            | pynutil.add_weight(implicit_mult, -0.22)
            | pynutil.add_weight(implicit_mult_greek, -0.22)
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
