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

from nemo_text_processing.text_normalization.hi.graph_utils import (
    NEMO_DIGIT,
    NEMO_HI_DIGIT,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.hi.utils import get_abs_path

# Convert Arabic digits (0-9) to Hindi digits (०-९)
arabic_to_hindi_digit = pynini.string_map([
    ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
    ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९")
]).optimize()
arabic_to_hindi_number = pynini.closure(arabic_to_hindi_digit).optimize()

# Load math operations
math_operations = pynini.string_file(get_abs_path("data/math_operations.tsv"))


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
        
        # Support both Hindi and Arabic digits
        # Hindi digits input
        hindi_number_input = pynini.closure(NEMO_HI_DIGIT, 1)
        hindi_number_graph = pynini.compose(hindi_number_input, cardinal_graph).optimize()
        
        # Arabic digits input
        arabic_number_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_number_graph = pynini.compose(
            arabic_number_input,
            arabic_to_hindi_number @ cardinal_graph
        ).optimize()
        
        # Combined number graph
        number_graph = hindi_number_graph | arabic_number_graph

        # Optional space around operators
        optional_space = pynini.closure(NEMO_SPACE, 0, 1)
        delimiter = optional_space | pynutil.insert(" ")

        # Operators that can appear between numbers
        # Exclude : and / to avoid conflicts with time and dates
        operators = pynini.union("+", "-", "*", "=", "&", "^", "%", "$", "#", "@", "!", "<", ">", ",", "(", ")", "?")
        
        # Math expression: number operator number
        # Pattern: number [space] operator [space] number
        math_expression = (
            pynutil.insert("left: \"")
            + number_graph
            + pynutil.insert("\"")
            + delimiter
            + pynutil.insert("operator: \"")
            + (operators @ math_operations)
            + pynutil.insert("\"")
            + delimiter
            + pynutil.insert("right: \"")
            + number_graph
            + pynutil.insert("\"")
        )

        # Also support: number operator number operator number (for longer expressions)
        # This handles cases like "1+2+3"
        extended_math = (
            pynutil.insert("left: \"")
            + number_graph
            + pynutil.insert("\"")
            + delimiter
            + pynutil.insert("operator: \"")
            + (operators @ math_operations)
            + pynutil.insert("\"")
            + delimiter
            + pynutil.insert("middle: \"")
            + number_graph
            + pynutil.insert("\"")
            + delimiter
            + pynutil.insert("operator2: \"")
            + (operators @ math_operations)
            + pynutil.insert("\"")
            + delimiter
            + pynutil.insert("right: \"")
            + number_graph
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

        final_graph = math_expression | extended_math | operator_number | number_operator | standalone_operator
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

