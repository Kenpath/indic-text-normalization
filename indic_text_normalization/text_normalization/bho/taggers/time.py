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

from indic_text_normalization.text_normalization.bho.graph_utils import (
    NEMO_BHO_ZERO,
    NEMO_DIGIT,
    NEMO_BHO_DIGIT,
    NEMO_BHO_NON_ZERO,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from indic_text_normalization.text_normalization.bho.utils import get_abs_path

# Time patterns specific to time tagger
BHO_DOUBLE_ZERO = "००"

# Convert Arabic digits (0-9) to Bhojpuri digits (०-९) - Devanagari script
arabic_to_bhojpuri_digit = pynini.string_map([
    ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
    ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९")
]).optimize()
arabic_to_bhojpuri_number = pynini.closure(arabic_to_bhojpuri_digit).optimize()

hours_graph = pynini.string_file(get_abs_path("data/time/hours.tsv"))
minutes_graph = pynini.string_file(get_abs_path("data/time/minutes.tsv"))
seconds_graph = pynini.string_file(get_abs_path("data/time/seconds.tsv"))


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        १२:३०:३०  -> time { hours: "बारह" minutes: "तीस" seconds: "तीस" }
        १:४०  -> time { hours: "एक" minutes: "चालीस" }
        १:००  -> time { hours: "एक" }
        9:15  -> time { hours: "नौ" minutes: "पंद्रह" }

    Args:
        time: GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="time", kind="classify")

        delete_colon = pynutil.delete(":")

        # Delete optional leading zero (handles inputs like 09, 07, 00)
        delete_leading_zero_bhojpuri = (
            (NEMO_BHO_NON_ZERO + NEMO_BHO_DIGIT)  # keep 10-99 as-is
            | (pynutil.delete(NEMO_BHO_ZERO) + NEMO_BHO_DIGIT)  # drop leading zero for 00-09
            | NEMO_BHO_DIGIT  # allow single-digit inputs
        ).optimize()
        delete_leading_zero_arabic = (
            (pynini.difference(NEMO_DIGIT, "0") + NEMO_DIGIT)  # keep 10-99 as-is
            | (pynutil.delete("0") + NEMO_DIGIT)  # drop leading zero for 00-09
            | NEMO_DIGIT  # allow single-digit inputs
        ).optimize()

        # Support both Bhojpuri and Arabic digits for hours and minutes
        # Bhojpuri digits path: delete optional leading zero -> hours_graph
        bhojpuri_hour_path = pynini.compose(delete_leading_zero_bhojpuri, hours_graph).optimize()
        # Arabic digits path: delete optional leading zero -> convert to Bhojpuri -> hours_graph
        arabic_hour_path = pynini.compose(
            delete_leading_zero_arabic,
            arabic_to_bhojpuri_number @ hours_graph
        ).optimize()
        hour_input = bhojpuri_hour_path | arabic_hour_path

        bhojpuri_minute_path = pynini.compose(pynini.closure(NEMO_BHO_DIGIT, 1), minutes_graph).optimize()
        arabic_minute_path = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_bhojpuri_number @ minutes_graph
        ).optimize()
        minute_input = bhojpuri_minute_path | arabic_minute_path

        bhojpuri_second_path = pynini.compose(pynini.closure(NEMO_BHO_DIGIT, 1), seconds_graph).optimize()
        arabic_second_path = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_bhojpuri_number @ seconds_graph
        ).optimize()
        second_input = bhojpuri_second_path | arabic_second_path

        self.hours = pynutil.insert("hours: \"") + hour_input + pynutil.insert("\" ")
        self.minutes = pynutil.insert("minutes: \"") + minute_input + pynutil.insert("\" ")
        self.seconds = pynutil.insert("seconds: \"") + second_input + pynutil.insert("\" ")

        # Optional "बजे" after time (to avoid duplication when verbalizer adds it)
        optional_baje = pynini.closure(
            pynini.closure(NEMO_SPACE, 0, 1) + pynutil.delete("बजे"), 0, 1
        ).optimize()

        # hour minute seconds - highest priority
        graph_hms = (
            self.hours + delete_colon + insert_space + self.minutes + delete_colon + insert_space + self.seconds + optional_baje
        )

        # hour minute - NORMAL FORMAT
        graph_hm = self.hours + delete_colon + insert_space + self.minutes + optional_baje

        # hour:00 format - support both Bhojpuri and Arabic double zeros
        arabic_double_zero = pynutil.delete("00")
        graph_h = (
            self.hours
            + delete_colon
            + (pynutil.delete(BHO_DOUBLE_ZERO) | arabic_double_zero)
            + optional_baje
        )

        # Prioritize: H:M:S > H:M > H:00
        final_graph = (
            pynutil.add_weight(graph_hms, -0.1)  # Highest priority
            | pynutil.add_weight(graph_hm, -0.05)  # Second priority
            | graph_h  # Third priority
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

