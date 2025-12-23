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

from indic_text_normalization.bn.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space, insert_space


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.
        time { hours: "বারো"  minutes: "দশ"  seconds: "দশ" } -> বারো বাজে দশ মিনিট দশ সেকেন্ড
        time { hours: "সাত" minutes: "চল্লিশ" } -> সাত বাজে চল্লিশ মিনিট
        time { hours: "দশ" } -> দশটা
        time { morphosyntactic_features: "দেড়" } -> দেড়টা

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

        insert_minute = pynutil.insert("মিনিট")
        insert_second = pynutil.insert("সেকেন্ড")
        insert_baje = pynutil.insert("বাজে")
        insert_ta = pynutil.insert("টা")

        # hour minute second - Format: hours + "বাজে" + minutes + "মিনিট" + seconds + "সেকেন্ড"
        graph_hms = (
            hour
            + delete_space
            + insert_baje
            + insert_space
            + minute
            + delete_space
            + insert_minute
            + insert_space
            + second
            + delete_space
            + insert_second
        )

        # Special time expressions (morphosyntactic_features) - Format: dedh/dhai/savva/sadhe/paune + "টা"
        graph_quarter = (
            pynutil.delete("morphosyntactic_features: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        )

        # hour minute - Format: hours + "বাজে" + minutes + "মিনিট"
        graph_hm = hour + delete_space + insert_baje + insert_space + minute + delete_space + insert_minute

        # hour - Format: hours + "টা"
        graph_h = hour + delete_space + insert_ta

        self.graph = graph_hms | graph_hm | graph_h | graph_quarter

        final_graph = self.graph

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()

