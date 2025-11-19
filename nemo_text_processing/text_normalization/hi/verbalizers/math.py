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

from nemo_text_processing.text_normalization.hi.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space, insert_space


class MathFst(GraphFst):
    """
    Finite state transducer for verbalizing math expressions, e.g.
        math { left: "एक" operator: "बराबर" right: "दो" } -> एक बराबर दो
        math { left: "एक" operator: "प्लस" right: "दो" } -> एक प्लस दो

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="math", kind="verbalize", deterministic=deterministic)

        left = (
            pynutil.delete("left:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 0)  # Allow empty for operator_number pattern
            + pynutil.delete("\"")
        )

        operator = (
            delete_space
            + pynutil.delete("operator:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        right = (
            delete_space
            + pynutil.delete("right:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 0)  # Allow empty for number_operator pattern
            + pynutil.delete("\"")
        )

        # Handle extended expressions (1+2+3)
        middle = (
            delete_space
            + pynutil.delete("middle:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        operator2 = (
            delete_space
            + pynutil.delete("operator2:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # Simple expression: left operator right
        simple_expression = (
            left + insert_space + operator + insert_space + right
        )

        # Extended expression: left operator middle operator2 right
        extended_expression = (
            left + insert_space + operator + insert_space + middle + insert_space + operator2 + insert_space + right
        )

        # Operator with number (e.g., "+5" -> "प्लस पांच")
        operator_number_expression = operator + insert_space + right

        # Number with operator (e.g., "5*" -> "पांच गुणा")
        number_operator_expression = left + insert_space + operator

        # Standalone operator (e.g., "+" -> "प्लस", "?" -> "प्रश्न चिह्न")
        # When both left and right are empty, just output the operator
        # This matches when left="" and right="", so we just output operator
        standalone_operator_expression = operator

        graph = (
            simple_expression 
            | extended_expression 
            | operator_number_expression 
            | number_operator_expression 
            | standalone_operator_expression
        )
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()

