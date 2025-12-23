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

from indic_text_normalization.te.graph_utils import (
    TE_DEDH,
    TE_DHAI,
    TE_PAUNE,
    TE_SADHE,
    TE_SAVVA,
    NEMO_DIGIT,
    NEMO_TE_DIGIT,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from indic_text_normalization.te.utils import get_abs_path

# Time patterns specific to time tagger
TE_DOUBLE_ZERO = "౦౦"
TE_TIME_FIFTEEN = ":౧౫"  # :15
TE_TIME_THIRTY = ":౩౦"  # :30
TE_TIME_FORTYFIVE = ":౪౫"  # :45

# Arabic time patterns
AR_TIME_FIFTEEN = ":15"
AR_TIME_THIRTY = ":30"
AR_TIME_FORTYFIVE = ":45"

# Convert Arabic digits (0-9) to Telugu digits (౦-౯)
arabic_to_telugu_digit = pynini.string_map([
    ("0", "౦"), ("1", "౧"), ("2", "౨"), ("3", "౩"), ("4", "౪"),
    ("5", "౫"), ("6", "౬"), ("7", "౭"), ("8", "౮"), ("9", "౯")
]).optimize()

# Create a converter for exactly 2 digits (for minutes/seconds)
# This ensures "40" -> "౪౦" (exactly 2 digits)
arabic_to_telugu_two_digits = (
    arabic_to_telugu_digit + arabic_to_telugu_digit
).optimize()

# For hours (1-2 digits), use closure
arabic_to_telugu_number = pynini.closure(arabic_to_telugu_digit).optimize()

hours_graph = pynini.string_file(get_abs_path("data/time/hours.tsv"))
minutes_graph = pynini.string_file(get_abs_path("data/time/minutes.tsv"))
seconds_graph = pynini.string_file(get_abs_path("data/time/seconds.tsv"))


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        ౧౨:౩౦:౩౦  -> time { hours: "పన్నెండు" minutes: "ముప్పై" seconds: "ముప్పై" }
        ౧:౪౦  -> time { hours: "ఒకటి" minutes: "నలభై" }
        ౧:౦౦  -> time { hours: "ఒకటి" }

    Args:
        time: GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="time", kind="classify")

        delete_colon = pynutil.delete(":")
        cardinal_graph = cardinal.digit | cardinal.teens_and_ties

        # Support both Telugu and Arabic digits for hours (1-2 digits: 0-23)
        # Telugu digits path: Telugu digits -> hours_graph
        telugu_hour_path = pynini.compose(pynini.closure(NEMO_TE_DIGIT, 1, 2), hours_graph).optimize()
        # Arabic digits path: Arabic digits -> convert to Telugu -> hours_graph
        arabic_hour_path = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1, 2), 
            arabic_to_telugu_number @ hours_graph
        ).optimize()
        hour_input = telugu_hour_path | arabic_hour_path

        # Minutes: support both 1 and 2 digits (0-59)
        # For 2 digits, use minutes_graph directly
        telugu_minute_two = pynini.compose(
            pynini.closure(NEMO_TE_DIGIT, 2, 2), 
            minutes_graph
        ).optimize()
        # For 1 digit, convert to Telugu and use cardinal
        telugu_minute_one = pynini.compose(
            pynini.closure(NEMO_TE_DIGIT, 1, 1),
            cardinal_graph
        ).optimize()
        telugu_minute_path = telugu_minute_two | telugu_minute_one
        
        # Arabic digits: exactly 2 digits, convert to Telugu, then match
        # Use the 2-digit converter to ensure proper conversion
        arabic_minute_two = pynini.compose(
            pynini.closure(NEMO_DIGIT, 2, 2),
            arabic_to_telugu_two_digits @ minutes_graph
        ).optimize()
        arabic_minute_one = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1, 1),
            arabic_to_telugu_number @ cardinal_graph
        ).optimize()
        arabic_minute_path = arabic_minute_two | arabic_minute_one
        minute_input = telugu_minute_path | arabic_minute_path

        # Seconds: support both 1 and 2 digits (0-59)
        # For 2 digits, use seconds_graph directly
        telugu_second_two = pynini.compose(
            pynini.closure(NEMO_TE_DIGIT, 2, 2), 
            seconds_graph
        ).optimize()
        # For 1 digit, convert to Telugu and use cardinal
        telugu_second_one = pynini.compose(
            pynini.closure(NEMO_TE_DIGIT, 1, 1),
            cardinal_graph
        ).optimize()
        telugu_second_path = telugu_second_two | telugu_second_one
        
        # Arabic digits: exactly 2 digits, convert to Telugu, then match
        # Use the 2-digit converter to ensure proper conversion
        arabic_second_two = pynini.compose(
            pynini.closure(NEMO_DIGIT, 2, 2),
            arabic_to_telugu_two_digits @ seconds_graph
        ).optimize()
        arabic_second_one = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1, 1),
            arabic_to_telugu_number @ cardinal_graph
        ).optimize()
        arabic_second_path = arabic_second_two | arabic_second_one
        second_input = telugu_second_path | arabic_second_path

        self.hours = pynutil.insert("hours: \"") + hour_input + pynutil.insert("\" ")
        self.minutes = pynutil.insert("minutes: \"") + minute_input + pynutil.insert("\" ")
        self.seconds = pynutil.insert("seconds: \"") + second_input + pynutil.insert("\" ")

        # hour minute seconds
        graph_hms = (
            self.hours + delete_colon + insert_space + self.minutes + delete_colon + insert_space + self.seconds
        )

        # hour minute
        graph_hm = self.hours + delete_colon + insert_space + self.minutes

        # hour - support both Telugu and Arabic double zero
        telugu_double_zero = pynutil.delete(TE_DOUBLE_ZERO)
        arabic_double_zero = pynutil.delete("00")
        double_zero = telugu_double_zero | arabic_double_zero
        graph_h = self.hours + delete_colon + double_zero

        # Support both Telugu and Arabic time patterns for dedh/dhai
        dedh_dhai_graph = (
            pynini.string_map([("౧" + TE_TIME_THIRTY, TE_DEDH), ("౨" + TE_TIME_THIRTY, TE_DHAI)])
            | pynini.string_map([("1" + AR_TIME_THIRTY, TE_DEDH), ("2" + AR_TIME_THIRTY, TE_DHAI)])
        )

        # Support both Telugu and Arabic time patterns
        # Fix: Only match :15 for savva, not :30
        savva_numbers = (
            (cardinal_graph + pynini.cross(TE_TIME_FIFTEEN, ""))
            | (cardinal_graph + pynini.cross(AR_TIME_FIFTEEN, ""))
        )
        savva_graph = pynutil.insert(TE_SAVVA) + pynutil.insert(NEMO_SPACE) + savva_numbers

        sadhe_numbers = (
            (cardinal_graph + pynini.cross(TE_TIME_THIRTY, ""))
            | (cardinal_graph + pynini.cross(AR_TIME_THIRTY, ""))
        )
        sadhe_graph = pynutil.insert(TE_SADHE) + pynutil.insert(NEMO_SPACE) + sadhe_numbers

        paune = pynini.string_file(get_abs_path("data/whitelist/paune_mappings.tsv"))
        paune_numbers = (
            (paune + pynini.cross(TE_TIME_FORTYFIVE, ""))
            | (paune + pynini.cross(AR_TIME_FORTYFIVE, ""))
        )
        paune_graph = pynutil.insert(TE_PAUNE) + pynutil.insert(NEMO_SPACE) + paune_numbers

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

        # Prioritize regular time patterns over special patterns (savva, sadhe, etc.)
        # to avoid incorrect matching like "౧౨:౩౦" being matched as "savva"
        # Lower weight = higher priority in pynini
        final_graph = (
            pynutil.add_weight(graph_hms, -1.0)  # Highest priority: H:MM:SS
            | pynutil.add_weight(graph_hm, -0.8)  # High priority: H:MM
            | pynutil.add_weight(graph_h, -0.6)  # Medium priority: H:00
            | pynutil.add_weight(graph_dedh_dhai, 0.1)  # Special patterns
            | pynutil.add_weight(graph_savva, 0.2)
            | pynutil.add_weight(graph_sadhe, 0.2)
            | pynutil.add_weight(graph_paune, 0.1)
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

