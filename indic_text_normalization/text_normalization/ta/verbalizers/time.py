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

from indic_text_normalization.text_normalization.ta.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space, insert_space


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.
        time { hours: "பன்னிரண்டு"  minutes: "பத்து"  seconds: "பத்து" } -> பன்னிரண்டு மணிக்கு பத்து நிமிடம் பத்து வினாடி
        time { hours: "ஏழு" minutes: "நாற்பது"" } -> ஏழு மணிக்கு நாற்பது நிமிடம்
        time { hours: "பத்து" } -> பத்து மணி

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="time", kind="verbalize")

        hour = pynutil.delete("hours: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"") + insert_space

        minute = (
            pynutil.delete("minutes: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"") + insert_space
        )

        second = (
            pynutil.delete("seconds: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"") + insert_space
        )

        insert_minute = pynutil.insert("நிமிடம்")
        insert_second = pynutil.insert("வினாடி")
        insert_manikku = pynutil.insert("மணிக்கு")
        insert_mani = pynutil.insert("மணி")

        # hour minute second - Format: hours + minutes + "நிமிடம்" + seconds + "வினாடி" + "மணிக்கு"
        graph_hms = (
            hour
            + delete_space
            + minute
            + delete_space
            + insert_minute
            + insert_space
            + second
            + delete_space
            + insert_second
            + insert_space
            + insert_manikku
        )

        graph_quarter = (
            pynutil.delete("morphosyntactic_features: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        )

        # hour minute - Format: hours + minutes + "நிமிடம்" + "மணிக்கு"
        graph_hm = hour + delete_space + minute + delete_space + insert_minute + insert_space + insert_manikku

        # hour - Format: hours + "மணி"
        graph_h = hour + delete_space + insert_mani

        self.graph = graph_hms | graph_hm | graph_h | graph_quarter

        final_graph = self.graph

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()

