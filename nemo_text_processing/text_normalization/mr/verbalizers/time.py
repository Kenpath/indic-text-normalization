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

from nemo_text_processing.text_normalization.mr.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space, insert_space


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.
        time { hours: "बारा"  minutes: "दहा"  seconds: "दहा" } -> बारा वाजून दहा मिनिटे दहा सेकंद
        time { hours: "सात" minutes: "चाळीस"" } -> सात वाजून चाळीस मिनिटे
        time { hours: "दहा" } -> दहा
        time { hours: "बारा" minutes: "शून्य" seconds: "शून्य" } -> बारा


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

        insert_minute = pynutil.insert("मिनिटे")
        insert_second = pynutil.insert("सेकंद")
        insert_vajun = pynutil.insert("वाजून")
        insert_vajta = pynutil.insert("वाजता")

        # hour minute second
        graph_hms = (
            hour
            + delete_space
            + insert_vajun
            + insert_space
            + minute
            + delete_space
            + insert_minute
            + insert_space
            + second
            + delete_space
            + insert_second
        )

        graph_quarter = (
            pynutil.delete("morphosyntactic_features: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        )

        # hour minute
        graph_hm = hour + delete_space + insert_vajun + insert_space + minute + delete_space + insert_minute

        # hour
        graph_h = hour + delete_space + insert_vajta

        self.graph = graph_hms | graph_hm | graph_h | graph_quarter

        final_graph = self.graph

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()

