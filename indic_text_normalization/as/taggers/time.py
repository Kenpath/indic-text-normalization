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

from ..graph_utils import (
    HI_DEDH,
    HI_DHAI,
    HI_PAUNE,
    HI_SADHE,
    HI_SAVVA,
    NEMO_DIGIT,
    NEMO_HI_DIGIT,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from ..utils import get_abs_path

# Time patterns specific to time tagger
HI_DOUBLE_ZERO = "००"
HI_TIME_FIFTEEN = ":१५"  # :15
HI_TIME_THIRTY = ":३०"  # :30
HI_TIME_FORTYFIVE = ":४५"  # :45

# Arabic time patterns
AR_TIME_FIFTEEN = ":15"
AR_TIME_THIRTY = ":30"
AR_TIME_FORTYFIVE = ":45"

# Convert Arabic digits (0-9) to Hindi digits (०-९)
arabic_to_hindi_digit = pynini.string_map([
    ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
    ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९")
]).optimize()
arabic_to_hindi_number = pynini.closure(arabic_to_hindi_digit).optimize()

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

        # Delete leading zeros from double-digit numbers (like English)
        # Pattern matches: "01"->"1", "02"->"2", ..., "09"->"9", "00"->"0", "12"->"12", "9"->"9"
        # For Arabic: handle all leading zero cases (01-09) by deleting leading "0", keep "12" as is
        # This matches English pattern: (NEMO_DIGIT + NEMO_DIGIT) | (pynini.closure(pynutil.delete("0"), 0, 1) + NEMO_DIGIT)
        delete_leading_zero_arabic = (
            # Match two digits (handles "00", "01", "02", ..., "09", "10", ..., "23")
            NEMO_DIGIT + NEMO_DIGIT
        ) | (
            # Match optional leading "0" (0 or 1 times) + digit
            # This handles: "0" + digit (like "09" -> "9") OR just digit (like "9" -> "9")
            pynini.closure(pynutil.delete("0"), 0, 1) + NEMO_DIGIT
        )
        # For Hindi: same pattern with Hindi digits
        delete_leading_zero_hindi = (
            # Match two Hindi digits
            NEMO_HI_DIGIT + NEMO_HI_DIGIT
        ) | (
            # Match optional leading "०" (0 or 1 times) + Hindi digit
            pynini.closure(pynutil.delete("०"), 0, 1) + NEMO_HI_DIGIT
        )

        # Support both Hindi and Arabic digits for hours and minutes
        # For hours: delete leading zeros, then convert to Hindi if needed, then map to hours_graph
        # Hindi digits path: Hindi digits (with leading zero deletion) -> hours_graph
        hindi_hour_path = pynini.compose(
            delete_leading_zero_hindi,
            hours_graph
        ).optimize()
        # Arabic digits path: delete leading zero -> convert to Hindi -> hours_graph
        arabic_hour_path = pynini.compose(
            delete_leading_zero_arabic,
            arabic_to_hindi_number @ hours_graph
        ).optimize()
        hour_input = hindi_hour_path | arabic_hour_path

        # For minutes: same approach
        hindi_minute_path = pynini.compose(
            delete_leading_zero_hindi,
            minutes_graph
        ).optimize()
        arabic_minute_path = pynini.compose(
            delete_leading_zero_arabic,
            arabic_to_hindi_number @ minutes_graph
        ).optimize()
        minute_input = hindi_minute_path | arabic_minute_path

        # For seconds: same approach
        hindi_second_path = pynini.compose(
            delete_leading_zero_hindi,
            seconds_graph
        ).optimize()
        arabic_second_path = pynini.compose(
            delete_leading_zero_arabic,
            arabic_to_hindi_number @ seconds_graph
        ).optimize()
        second_input = hindi_second_path | arabic_second_path

        self.hours = pynutil.insert("hours: \"") + hour_input + pynutil.insert("\" ")
        self.minutes = pynutil.insert("minutes: \"") + minute_input + pynutil.insert("\" ")
        self.seconds = pynutil.insert("seconds: \"") + second_input + pynutil.insert("\" ")

        # hour minute seconds
        graph_hms = (
            self.hours + delete_colon + insert_space + self.minutes + delete_colon + insert_space + self.seconds
        )

        # hour minute
        graph_hm = self.hours + delete_colon + insert_space + self.minutes

        # hour - support both Hindi and Arabic double zero
        hindi_double_zero = pynutil.delete(HI_DOUBLE_ZERO)
        arabic_double_zero = pynutil.delete("00")
        double_zero = hindi_double_zero | arabic_double_zero
        graph_h = self.hours + delete_colon + double_zero

        # Support both Hindi and Arabic time patterns for dedh/dhai
        dedh_dhai_graph = (
            pynini.string_map([("१" + HI_TIME_THIRTY, HI_DEDH), ("२" + HI_TIME_THIRTY, HI_DHAI)])
            | pynini.string_map([("1" + AR_TIME_THIRTY, HI_DEDH), ("2" + AR_TIME_THIRTY, HI_DHAI)])
        )

        # Support both Hindi and Arabic time patterns
        savva_numbers = (
            (cardinal_graph + pynini.cross(HI_TIME_FIFTEEN, ""))
            | (cardinal_graph + pynini.cross(AR_TIME_FIFTEEN, ""))
        )
        savva_graph = pynutil.insert(HI_SAVVA) + pynutil.insert(NEMO_SPACE) + savva_numbers

        sadhe_numbers = (
            (cardinal_graph + pynini.cross(HI_TIME_THIRTY, ""))
            | (cardinal_graph + pynini.cross(AR_TIME_THIRTY, ""))
        )
        sadhe_graph = pynutil.insert(HI_SADHE) + pynutil.insert(NEMO_SPACE) + sadhe_numbers

        paune = pynini.string_file(get_abs_path("data/whitelist/paune_mappings.tsv"))
        paune_numbers = (
            (paune + pynini.cross(HI_TIME_FORTYFIVE, ""))
            | (paune + pynini.cross(AR_TIME_FORTYFIVE, ""))
        )
        paune_graph = pynutil.insert(HI_PAUNE) + pynutil.insert(NEMO_SPACE) + paune_numbers

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

        final_graph = (
            graph_hms
            | pynutil.add_weight(graph_hm, 0.3)
            | pynutil.add_weight(graph_h, 0.3)
            | pynutil.add_weight(graph_dedh_dhai, 0.1)
            | pynutil.add_weight(graph_savva, 0.2)
            | pynutil.add_weight(graph_sadhe, 0.2)
            | pynutil.add_weight(graph_paune, 0.1)
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
