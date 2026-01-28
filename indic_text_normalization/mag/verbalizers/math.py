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

from indic_text_normalization.mag.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space, insert_space


class MathFst(GraphFst):
    """
    Finite state transducer for verbalizing math expressions, e.g.
        math { left: "एक" operator: "बराबर" right: "दुइ" } -> एक बराबर दुइ
        math { left: "एक" operator: "जोड़" right: "दुइ" } -> एक जोड़ दुइ

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
            + pynini.closure(NEMO_NOT_QUOTE, 0)  # allow empty (operator_number / standalone)
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
            + pynini.closure(NEMO_NOT_QUOTE, 0)  # allow empty (number_operator / standalone)
            + pynutil.delete("\"")
        )

        # Extended expressions (needed for tight patterns like "10-2=8")
        middle = (
            delete_space
            + pynutil.delete("middle:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        operator_two = (
            delete_space
            + pynutil.delete("operator_two:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # Simple expression: left operator right
        simple_expression = left + insert_space + operator + insert_space + right

        # Extended expression: left operator middle operator_two right
        extended_expression = (
            left
            + insert_space
            + operator
            + insert_space
            + middle
            + insert_space
            + operator_two
            + insert_space
            + right
        )

        # Operator with number (e.g., "+5" -> "प्लस पांच")
        operator_number_expression = operator + insert_space + right

        # Number with operator (e.g., "5*" -> "पांच गुणा")
        number_operator_expression = left + insert_space + operator

        # Standalone operator (e.g., "+")
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

