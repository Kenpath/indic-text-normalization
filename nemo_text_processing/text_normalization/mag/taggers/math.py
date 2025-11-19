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

from nemo_text_processing.text_normalization.mag.graph_utils import (
    NEMO_DIGIT,
    NEMO_MAG_DIGIT,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.mag.utils import get_abs_path

# Convert Arabic digits (0-9) to Magadhi digits (०-९)
arabic_to_magadhi_digit = pynini.string_map([
    ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
    ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९")
]).optimize()
arabic_to_magadhi_number = pynini.closure(arabic_to_magadhi_digit).optimize()

# Load math operations
math_operations = pynini.string_file(get_abs_path("data/math_operations.tsv"))


class MathFst(GraphFst):
    """
    Finite state transducer for classifying math expressions, e.g.
        "1=2" -> math { left: "एक" operator: "बराबर" right: "दुइ" }
        "1+2" -> math { left: "एक" operator: "जोड़" right: "दुइ" }
        "१२=३४" -> math { left: "बारह" operator: "बराबर" right: "चौंतीस" }

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="math", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph
        
        # Support both Magadhi and Arabic digits
        # Magadhi digits input
        magadhi_number_input = pynini.closure(NEMO_MAG_DIGIT, 1)
        magadhi_number_graph = pynini.compose(magadhi_number_input, cardinal_graph).optimize()
        
        # Arabic digits input
        arabic_number_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_number_graph = pynini.compose(
            arabic_number_input,
            arabic_to_magadhi_number @ cardinal_graph
        ).optimize()
        
        # Combined number graph
        number_graph = magadhi_number_graph | arabic_number_graph

        # Optional space around operators
        optional_space = pynini.closure(NEMO_SPACE, 0, 1)
        delimiter = optional_space | pynutil.insert(" ")

        # Operators that can appear between numbers
        # Exclude : and / to avoid conflicts with time and dates
        operators = pynini.union("+", "-", "*", "=", "&", "^", "%", "$", "#", "@", "!", "<", ">", ",", "(", ")")
        
        # Math expression: number operator number
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

        final_graph = math_expression
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

