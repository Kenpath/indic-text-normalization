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

from nemo_text_processing.text_normalization.bho.graph_utils import (
    NEMO_BHO_ZERO,
    NEMO_DIGIT,
    NEMO_BHO_DIGIT,
    BHO_DEDH,
    BHO_DHAI,
    BHO_PAUNE,
    BHO_SADHE,
    BHO_SAVVA,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.bho.utils import get_abs_path

# Time patterns specific to time tagger
BHO_DOUBLE_ZERO = "००"
BHO_TIME_FIFTEEN = ":१५"  # :15
BHO_TIME_THIRTY = ":३०"  # :30
BHO_TIME_FORTYFIVE = ":४५"  # :45

# Arabic time patterns
AR_TIME_FIFTEEN = ":15"
AR_TIME_THIRTY = ":30"
AR_TIME_FORTYFIVE = ":45"

# Convert Arabic digits (0-9) to Bhojpuri digits (०-९)
arabic_to_bhojpuri_digit = pynini.string_map([
    ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
    ("5", "५"), ("6", "६"), ("7", "७"), ("8", "৮"), ("9", "९")
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

    Args:
        time: GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="time", kind="classify")

        delete_colon = pynutil.delete(":")
        cardinal_graph = cardinal.digit | cardinal.teens_and_ties

        # Support both Bhojpuri and Arabic digits for hours and minutes
        # Create combined graphs that accept both Arabic and Bhojpuri digits
        # Bhojpuri digits path: Bhojpuri digits -> hours_graph
        bhojpuri_hour_path = pynini.compose(pynini.closure(NEMO_BHO_DIGIT, 1), hours_graph).optimize()
        # Arabic digits path: Arabic digits -> convert to Bhojpuri -> hours_graph
        arabic_hour_path = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1), 
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
        # Handle optional space(s) before "बजे"
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
        graph_h = self.hours + delete_colon + pynutil.delete(BHO_DOUBLE_ZERO) + optional_baje

        dedh_dhai_graph = pynini.string_map([("१" + BHO_TIME_THIRTY, BHO_DEDH), ("२" + BHO_TIME_THIRTY, BHO_DHAI)])

        savva_numbers = cardinal_graph + pynini.cross(BHO_TIME_FIFTEEN, "")
        savva_graph = pynutil.insert(BHO_SAVVA) + pynutil.insert(NEMO_SPACE) + savva_numbers

        sadhe_numbers = cardinal_graph + pynini.cross(BHO_TIME_THIRTY, "")
        sadhe_graph = pynutil.insert(BHO_SADHE) + pynutil.insert(NEMO_SPACE) + sadhe_numbers

        paune = pynini.string_file(get_abs_path("data/whitelist/paune_mappings.tsv"))
        paune_numbers = paune + pynini.cross(BHO_TIME_FORTYFIVE, "")
        paune_graph = pynutil.insert(BHO_PAUNE) + pynutil.insert(NEMO_SPACE) + paune_numbers

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

        # Prioritize normal hour:minute format over special Bhojpuri time expressions
        # Use very high weight for normal format, very low weights for special expressions
        final_graph = (
            graph_hms
            | pynutil.add_weight(graph_hm, 1.0)  # Highest priority for normal hour:minute format
            | pynutil.add_weight(graph_h, 0.8)
            | pynutil.add_weight(graph_dedh_dhai, 0.01)  # Very low weight - almost disabled
            | pynutil.add_weight(graph_savva, 0.01)  # Very low weight - almost disabled
            | pynutil.add_weight(graph_sadhe, 0.01)  # Very low weight - almost disabled
            | pynutil.add_weight(graph_paune, 0.01)  # Very low weight - almost disabled
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

