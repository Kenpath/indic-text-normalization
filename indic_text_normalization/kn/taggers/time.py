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

from indic_text_normalization.kn.graph_utils import (
    KN_DEDH,
    KN_DHAI,
    KN_PAUNE,
    KN_SADHE,
    KN_SAVVA,
    NEMO_DIGIT,
    NEMO_KN_DIGIT,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from indic_text_normalization.kn.utils import get_abs_path

# Time patterns specific to time tagger
KN_DOUBLE_ZERO = "೦೦"
KN_TIME_FIFTEEN = ":೧೫"  # :15
KN_TIME_THIRTY = ":೩೦"  # :30
KN_TIME_FORTYFIVE = ":೪೫"  # :45

# Arabic time patterns
AR_TIME_FIFTEEN = ":15"
AR_TIME_THIRTY = ":30"
AR_TIME_FORTYFIVE = ":45"

# Convert Arabic digits (0-9) to Kannada digits (೦-೯)
arabic_to_kannada_digit = pynini.string_map([
    ("0", "೦"), ("1", "೧"), ("2", "೨"), ("3", "೩"), ("4", "೪"),
    ("5", "೫"), ("6", "೬"), ("7", "೭"), ("8", "೮"), ("9", "೯")
]).optimize()

# Create a converter for exactly 2 digits (for minutes/seconds)
# This ensures "40" -> "೪೦" (exactly 2 digits)
arabic_to_kannada_two_digits = (
    arabic_to_kannada_digit + arabic_to_kannada_digit
).optimize()

# For hours (1-2 digits), use closure
arabic_to_kannada_number = pynini.closure(arabic_to_kannada_digit).optimize()

hours_graph = pynini.string_file(get_abs_path("data/time/hours.tsv"))
minutes_graph = pynini.string_file(get_abs_path("data/time/minutes.tsv"))
seconds_graph = pynini.string_file(get_abs_path("data/time/seconds.tsv"))


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        ೧೨:೩೦:೩೦  -> time { hours: "ಹನ್ನೆರಡು" minutes: "ಮೂವತ್ತು" seconds: "ಮೂವತ್ತು" }
        ೧:೪೦  -> time { hours: "ಒಂದು" minutes: "ನಲವತ್ತು" }
        ೧:೦೦  -> time { hours: "ಒಂದು" }

    Args:
        time: GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="time", kind="classify")

        delete_colon = pynutil.delete(":")
        cardinal_graph = cardinal.digit | cardinal.teens_and_ties

        # Support both Kannada and Arabic digits for hours (1-2 digits: 0-23)
        # Kannada digits path: Kannada digits -> hours_graph
        kannada_hour_path = pynini.compose(pynini.closure(NEMO_KN_DIGIT, 1, 2), hours_graph).optimize()
        # Arabic digits path: Arabic digits -> convert to Kannada -> hours_graph
        arabic_hour_path = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1, 2), 
            arabic_to_kannada_number @ hours_graph
        ).optimize()
        hour_input = kannada_hour_path | arabic_hour_path

        # Minutes: support both 1 and 2 digits (0-59)
        # For 2 digits, use minutes_graph directly
        kannada_minute_two = pynini.compose(
            pynini.closure(NEMO_KN_DIGIT, 2, 2), 
            minutes_graph
        ).optimize()
        # For 1 digit, convert to Kannada and use cardinal
        kannada_minute_one = pynini.compose(
            pynini.closure(NEMO_KN_DIGIT, 1, 1),
            cardinal_graph
        ).optimize()
        kannada_minute_path = kannada_minute_two | kannada_minute_one
        
        # Arabic digits: exactly 2 digits, convert to Kannada, then match
        # Use the 2-digit converter to ensure proper conversion
        arabic_minute_two = pynini.compose(
            pynini.closure(NEMO_DIGIT, 2, 2),
            arabic_to_kannada_two_digits @ minutes_graph
        ).optimize()
        arabic_minute_one = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1, 1),
            arabic_to_kannada_number @ cardinal_graph
        ).optimize()
        arabic_minute_path = arabic_minute_two | arabic_minute_one
        minute_input = kannada_minute_path | arabic_minute_path

        # Seconds: support both 1 and 2 digits (0-59)
        # For 2 digits, use seconds_graph directly
        kannada_second_two = pynini.compose(
            pynini.closure(NEMO_KN_DIGIT, 2, 2), 
            seconds_graph
        ).optimize()
        # For 1 digit, convert to Kannada and use cardinal
        kannada_second_one = pynini.compose(
            pynini.closure(NEMO_KN_DIGIT, 1, 1),
            cardinal_graph
        ).optimize()
        kannada_second_path = kannada_second_two | kannada_second_one
        
        # Arabic digits: exactly 2 digits, convert to Kannada, then match
        # Use the 2-digit converter to ensure proper conversion
        arabic_second_two = pynini.compose(
            pynini.closure(NEMO_DIGIT, 2, 2),
            arabic_to_kannada_two_digits @ seconds_graph
        ).optimize()
        arabic_second_one = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1, 1),
            arabic_to_kannada_number @ cardinal_graph
        ).optimize()
        arabic_second_path = arabic_second_two | arabic_second_one
        second_input = kannada_second_path | arabic_second_path

        self.hours = pynutil.insert("hours: \"") + hour_input + pynutil.insert("\" ")
        self.minutes = pynutil.insert("minutes: \"") + minute_input + pynutil.insert("\" ")
        self.seconds = pynutil.insert("seconds: \"") + second_input + pynutil.insert("\" ")

        # hour minute seconds
        graph_hms = (
            self.hours + delete_colon + insert_space + self.minutes + delete_colon + insert_space + self.seconds
        )

        # hour minute
        graph_hm = self.hours + delete_colon + insert_space + self.minutes

        # hour - support both Kannada and Arabic double zero
        kannada_double_zero = pynutil.delete(KN_DOUBLE_ZERO)
        arabic_double_zero = pynutil.delete("00")
        double_zero = kannada_double_zero | arabic_double_zero
        graph_h = self.hours + delete_colon + double_zero

        # Support both Kannada and Arabic time patterns for dedh/dhai
        dedh_dhai_graph = (
            pynini.string_map([("೧" + KN_TIME_THIRTY, KN_DEDH), ("೨" + KN_TIME_THIRTY, KN_DHAI)])
            | pynini.string_map([("1" + AR_TIME_THIRTY, KN_DEDH), ("2" + AR_TIME_THIRTY, KN_DHAI)])
        )

        # Support both Kannada and Arabic time patterns
        # Fix: Only match :15 for savva, not :30
        savva_numbers = (
            (cardinal_graph + pynini.cross(KN_TIME_FIFTEEN, ""))
            | (cardinal_graph + pynini.cross(AR_TIME_FIFTEEN, ""))
        )
        savva_graph = pynutil.insert(KN_SAVVA) + pynutil.insert(NEMO_SPACE) + savva_numbers

        sadhe_numbers = (
            (cardinal_graph + pynini.cross(KN_TIME_THIRTY, ""))
            | (cardinal_graph + pynini.cross(AR_TIME_THIRTY, ""))
        )
        sadhe_graph = pynutil.insert(KN_SADHE) + pynutil.insert(NEMO_SPACE) + sadhe_numbers

        paune = pynini.string_file(get_abs_path("data/whitelist/paune_mappings.tsv"))
        paune_numbers = (
            (paune + pynini.cross(KN_TIME_FORTYFIVE, ""))
            | (paune + pynini.cross(AR_TIME_FORTYFIVE, ""))
        )
        paune_graph = pynutil.insert(KN_PAUNE) + pynutil.insert(NEMO_SPACE) + paune_numbers

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
        # to avoid incorrect matching like "೧೨:೩೦" being matched as "savva"
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

