# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.mr.graph_utils import NEMO_NOT_QUOTE, NEMO_SPACE, GraphFst, delete_space


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date, e.g.
        date { day: "एक" month: "एप्रिल" year: "दोन हजार चोवीस" } -> "एक एप्रिल दोन हजार चोवीस"
        date { month: "एप्रिल" day: "एक" year: "दोन हजार चोवीस" } -> "एप्रिल एक दोन हजार चोवीस"


    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self):
        super().__init__(name="date", kind="verbalize")

        day = pynutil.delete("day: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        month = pynutil.delete("month: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        year = pynutil.delete("year: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        graph_era = pynutil.delete("era: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        graph_dd_mm = day + NEMO_SPACE + month

        graph_mm_dd = month + NEMO_SPACE + day

        graph_dd_mm_yyyy = day + NEMO_SPACE + month + NEMO_SPACE + year

        graph_mm_dd_yyyy = month + NEMO_SPACE + day + NEMO_SPACE + year

        graph_mm_yyyy = month + NEMO_SPACE + year

        optional_preserve_order = pynini.closure(
            pynutil.delete("preserve_order:") + delete_space + pynutil.delete("true") + delete_space
            | pynutil.delete("field_order:")
            + delete_space
            + pynutil.delete("\"")
            + NEMO_NOT_QUOTE
            + pynutil.delete("\"")
            + delete_space
        )

        self.graph = (
            (graph_dd_mm | graph_mm_dd | graph_dd_mm_yyyy | graph_mm_dd_yyyy | graph_mm_yyyy | graph_era)
            + delete_space
            + optional_preserve_order
        )

        final_graph = self.graph

        delete_tokens = self.delete_tokens(final_graph)

        self.fst = delete_tokens.optimize()

