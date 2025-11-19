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

from nemo_text_processing.text_normalization.mag.graph_utils import (
    NEMO_MAG_ZERO,
    NEMO_DIGIT,
    NEMO_MAG_DIGIT,
    MAG_DEDH,
    MAG_DHAI,
    MAG_PAUNE,
    MAG_SADHE,
    MAG_SAVVA,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.mag.utils import get_abs_path

# Time patterns specific to time tagger
MAG_DOUBLE_ZERO = "००"
MAG_TIME_FIFTEEN = ":१५"  # :15
MAG_TIME_THIRTY = ":३०"  # :30
MAG_TIME_FORTYFIVE = ":४५"  # :45

# Arabic time patterns
AR_TIME_FIFTEEN = ":15"
AR_TIME_THIRTY = ":30"
AR_TIME_FORTYFIVE = ":45"

# Convert Arabic digits (0-9) to Magadhi digits (०-९)
arabic_to_magadhi_digit = pynini.string_map([
    ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
    ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९")
]).optimize()
arabic_to_magadhi_number = pynini.closure(arabic_to_magadhi_digit).optimize()

hours_graph = pynini.string_file(get_abs_path("data/time/hours.tsv"))
minutes_graph = pynini.string_file(get_abs_path("data/time/minutes.tsv"))
seconds_graph = pynini.string_file(get_abs_path("data/time/seconds.tsv"))


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        १२:३०:३०  -> time { hours: "बारह" minutes: "तीस" seconds: "तीस" }
        १:४०  -> time { hours: "एक" minutes: "चालीस" }
        १:००  -> time { hours: "एक" }

    Args:
        time: GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="time", kind="classify")

        delete_colon = pynutil.delete(":")
        cardinal_graph = cardinal.digit | cardinal.teens_and_ties

        # Support both Magadhi and Arabic digits for hours and minutes
        # Magadhi digits path: Magadhi digits -> hours_graph
        magadhi_hour_path = pynini.compose(pynini.closure(NEMO_MAG_DIGIT, 1), hours_graph).optimize()
        # Arabic digits path: Arabic digits -> convert to Magadhi -> hours_graph
        arabic_hour_path = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1), 
            arabic_to_magadhi_number @ hours_graph
        ).optimize()
        hour_input = magadhi_hour_path | arabic_hour_path

        magadhi_minute_path = pynini.compose(pynini.closure(NEMO_MAG_DIGIT, 1), minutes_graph).optimize()
        arabic_minute_path = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_magadhi_number @ minutes_graph
        ).optimize()
        minute_input = magadhi_minute_path | arabic_minute_path

        magadhi_second_path = pynini.compose(pynini.closure(NEMO_MAG_DIGIT, 1), seconds_graph).optimize()
        arabic_second_path = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_magadhi_number @ seconds_graph
        ).optimize()
        second_input = magadhi_second_path | arabic_second_path

        self.hours = pynutil.insert("hours: \"") + hour_input + pynutil.insert("\" ")
        self.minutes = pynutil.insert("minutes: \"") + minute_input + pynutil.insert("\" ")
        self.seconds = pynutil.insert("seconds: \"") + second_input + pynutil.insert("\" ")

        # Optional "बजे" after time (to avoid duplication when verbalizer adds it)
        optional_baje = pynini.closure(
            pynini.closure(NEMO_SPACE, 0, 1) + pynutil.delete("बजे"), 0, 1
        ).optimize()

        # hour minute seconds
        graph_hms = (
            self.hours + delete_colon + insert_space + self.minutes + delete_colon + insert_space + self.seconds + optional_baje
        )

        # hour minute - NORMAL FORMAT (highest priority)
        graph_hm = self.hours + delete_colon + insert_space + self.minutes + optional_baje

        # hour
        graph_h = self.hours + delete_colon + pynutil.delete(MAG_DOUBLE_ZERO) + optional_baje

        dedh_dhai_graph = pynini.string_map([("१" + MAG_TIME_THIRTY, MAG_DEDH), ("२" + MAG_TIME_THIRTY, MAG_DHAI)])

        savva_numbers = cardinal_graph + pynini.cross(MAG_TIME_FIFTEEN, "")
        savva_graph = pynutil.insert(MAG_SAVVA) + pynutil.insert(NEMO_SPACE) + savva_numbers

        sadhe_numbers = cardinal_graph + pynini.cross(MAG_TIME_THIRTY, "")
        sadhe_graph = pynutil.insert(MAG_SADHE) + pynutil.insert(NEMO_SPACE) + sadhe_numbers

        paune = pynini.string_file(get_abs_path("data/whitelist/paune_mappings.tsv"))
        paune_numbers = paune + pynini.cross(MAG_TIME_FORTYFIVE, "")
        paune_graph = pynutil.insert(MAG_PAUNE) + pynutil.insert(NEMO_SPACE) + paune_numbers

        graph_dedh_dhai = (
            pynutil.insert("morphosyntactic_features: \"")
            + dedh_dhai_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        graph_savva = (
            pynutil.insert("morphosyntactic_features: \"")
            + savva_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        graph_sadhe = (
            pynutil.insert("morphosyntactic_features: \"")
            + sadhe_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        graph_paune = (
            pynutil.insert("morphosyntactic_features: \"")
            + paune_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        # Prioritize normal hour:minute format over special Magadhi time expressions
        final_graph = (
            graph_hms
            | pynutil.add_weight(graph_hm, 1.0)  # Highest priority for normal hour:minute format
            | pynutil.add_weight(graph_h, 0.8)
            | pynutil.add_weight(graph_dedh_dhai, 0.01)  # Very low weight
            | pynutil.add_weight(graph_savva, 0.01)  # Very low weight
            | pynutil.add_weight(graph_sadhe, 0.01)  # Very low weight
            | pynutil.add_weight(graph_paune, 0.01)  # Very low weight
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

