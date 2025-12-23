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

from indic_text_normalization.te.graph_utils import NEMO_CHAR, GraphFst, delete_space, insert_space


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimal, e.g.
        decimal { negative: "true" integer_part: "12" fractional_part: "5006" quantity: "billion" } -> minus twelve point five zero zero six billion

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="decimal", kind="verbalize", deterministic=deterministic)

        optional_sign = pynini.closure(
            pynini.cross("negative: \"true\"", "minus ") + delete_space, 0, 1
        )

        integer = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(pynini.difference(pynini.union(pynini.closure(NEMO_CHAR), " "), "\""), 1)
            + pynutil.delete("\"")
        )

        fractional = (
            pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(pynini.difference(pynini.union(pynini.closure(NEMO_CHAR), " "), "\""), 1)
            + pynutil.delete("\"")
        )

        quantity = (
            delete_space
            + insert_space
            + pynutil.delete("quantity:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(pynini.difference(pynini.union(pynini.closure(NEMO_CHAR), " "), "\""), 1)
            + pynutil.delete("\"")
        )

        graph = optional_sign + integer
        graph += delete_space + pynutil.insert(" point ") + delete_space + fractional
        graph += pynini.closure(quantity, 0, 1)

        self.numbers = graph
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()

