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

from nemo_text_processing.text_normalization.te.graph_utils import (
    NEMO_DIGIT,
    NEMO_TE_DIGIT,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.te.utils import get_abs_path

# Convert Arabic digits (0-9) to Telugu digits (౦-౯)
arabic_to_telugu_digit = pynini.string_map([
    ("0", "౦"), ("1", "౧"), ("2", "౨"), ("3", "౩"), ("4", "౪"),
    ("5", "౫"), ("6", "౬"), ("7", "౭"), ("8", "౮"), ("9", "౯")
]).optimize()
arabic_to_telugu_number = pynini.closure(arabic_to_telugu_digit).optimize()

# Load math operations
math_operations = pynini.string_file(get_abs_path("data/math_operations.tsv"))


class MathFst(GraphFst):
    """
    Finite state transducer for classifying math expressions, e.g.
        "1=2" -> math { left: "ఒకటి" operator: "సమానం" right: "రెండు" }
        "1+2" -> math { left: "ఒకటి" operator: "ప్లస్" right: "రెండు" }
        "౧౨=౩౪" -> math { left: "పన్నెండు" operator: "సమానం" right: "ముప్పై నాలుగు" }

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="math", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph
        
        # Support both Telugu and Arabic digits
        # Telugu digits input
        telugu_number_input = pynini.closure(NEMO_TE_DIGIT, 1)
        telugu_number_graph = pynini.compose(telugu_number_input, cardinal_graph).optimize()
        
        # Arabic digits input
        arabic_number_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_number_graph = pynini.compose(
            arabic_number_input,
            arabic_to_telugu_number @ cardinal_graph
        ).optimize()
        
        # Combined number graph
        number_graph = telugu_number_graph | arabic_number_graph

        # Optional space around operators
        optional_space = pynini.closure(NEMO_SPACE, 0, 1)
        delimiter = optional_space | pynutil.insert(" ")

        # Operators that can appear between numbers
        # Exclude : and / to avoid conflicts with time and dates
        operators = pynini.union("+", "-", "*", "=", "&", "^", "%", "$", "#", "@", "!", "<", ">", ",", "(", ")")
        
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

        final_graph = math_expression | extended_math
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

