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

from indic_text_normalization.bho.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space, insert_space


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measure, e.g.
        measure { negative: "true" cardinal { integer: "बारह" } units: "किलोग्राम" } -> ऋण बारह किलोग्राम
        measure { decimal { integer_part: "बारह" fractional_part: "दो" } units: "किलोग्राम" } -> बारह दशमलव दो किलोग्राम


    Args:
        decimal: DecimalFst
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="measure", kind="verbalize", deterministic=deterministic)

        optional_sign = pynini.closure(
            pynini.cross("negative: \"true\" ", " ऋण "),  # "negative" in Bhojpuri/Hindi
            0,
            1,
        )

        unit = pynutil.delete("units: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"") + delete_space

        # Handle integer part
        integer = (
            pynutil.delete("integer: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # Handle decimal parts
        integer_part = (
            pynutil.delete("integer_part: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        fractional_part = (
            pynutil.delete("fractional_part: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # Decimal graph
        graph_decimal = (
            pynutil.delete("decimal {")
            + delete_space
            + optional_sign
            + integer_part
            + delete_space
            + pynutil.insert(" दशमलव ")  # "decimal point" in Bhojpuri/Hindi
            + fractional_part
            + delete_space
            + pynutil.delete("}")
        )

        # Cardinal graph
        graph_cardinal = (
            pynutil.delete("cardinal {")
            + delete_space
            + optional_sign
            + integer
            + delete_space
            + pynutil.delete("}")
        )

        graph = (graph_cardinal | graph_decimal) + delete_space + insert_space + unit
        self.decimal = graph_decimal
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()

