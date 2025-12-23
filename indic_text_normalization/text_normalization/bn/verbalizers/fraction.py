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

from indic_text_normalization.text_normalization.bn.graph_utils import (
    NEMO_NOT_QUOTE, 
    GraphFst, 
    insert_space,
)


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fractions, e.g.
        fraction { numerator: "তিন" denominator: "চার" } -> তিন ভাগ চার
        fraction { integer_part: "তেইশ" numerator: "চার" denominator: "ছয়" } -> তেইশ এবং চার ভাগ ছয়
    
    Following English/Tamil fraction verbalizer pattern.
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

        # Integer part extraction
        integer = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\" ")

        # Numerator
        numerator = (
            pynutil.delete("numerator: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\" ")
        )

        # Denominator
        denominator = (
            pynutil.delete("denominator: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # Regular fraction: numerator + "ভাগ" + denominator
        graph = numerator + pynutil.insert("ভাগ ") + denominator

        # Add "এবং" (and) for mixed numbers
        conjunction = pynutil.insert("এবং ")
        integer = pynini.closure(integer + conjunction, 0, 1)

        graph = integer + graph

        self.graph = graph
        delete_tokens = self.delete_tokens(self.graph)
        self.fst = delete_tokens.optimize()

