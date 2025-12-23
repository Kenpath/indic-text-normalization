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

from indic_text_normalization.ta.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
    insert_space,
)


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measure, e.g.
        measure { negative: "true" cardinal { integer: "பன்னிரண்டு" } units: "கிலோகிராம்" } -> கழித்தல் பன்னிரண்டு கிலோகிராம்
        measure { decimal { integer_part: "பன்னிரண்டு" fractional_part: "ஐந்து" } units: "கிலோகிராம்" } -> பன்னிரண்டு புள்ளி ஐந்து கிலோகிராம்

    Args:
        decimal: DecimalFst
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="measure", kind="verbalize", deterministic=deterministic)

        optional_graph_negative = pynini.closure(
            pynini.cross("negative: \"true\" ", " கழித்தல் "),
            0,
            1,
        )

        unit = (
            pynutil.delete("units: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + delete_space
        )

        # Cardinal verbalization
        graph_cardinal = (
            pynutil.delete("cardinal {")
            + delete_space
            + optional_graph_negative
            + pynutil.delete("integer: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + delete_space
            + pynutil.delete("}")
        )

        # Decimal verbalization
        graph_decimal = (
            pynutil.delete("decimal {")
            + delete_space
            + optional_graph_negative
            + decimal.numbers
            + delete_space
            + pynutil.delete("}")
        )

        # Optional preserve_order handling
        optional_preserve_order = pynini.closure(
            pynutil.delete("preserve_order:") + delete_space + pynutil.delete("true") + delete_space
            | pynutil.delete("field_order:")
            + delete_space
            + pynutil.delete("\"")
            + NEMO_NOT_QUOTE
            + pynutil.delete("\"")
            + delete_space,
            0,
            1
        )

        graph = (
            (graph_cardinal | graph_decimal)
            + delete_space
            + insert_space
            + unit
            + optional_preserve_order
        )

        self.graph = graph
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()

