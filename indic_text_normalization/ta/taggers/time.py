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

from indic_text_normalization.ta.graph_utils import (
    NEMO_TA_ZERO,
    NEMO_DIGIT,
    NEMO_TA_DIGIT,
    NEMO_TA_NON_ZERO,
    TA_DEDH,
    TA_DHAI,
    TA_PAUNE,
    TA_SADHE,
    TA_SAVVA,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from indic_text_normalization.ta.utils import get_abs_path

# Time patterns specific to time tagger
TA_DOUBLE_ZERO = "௦௦"
TA_TIME_FIFTEEN = ":௧௫"  # :15
TA_TIME_THIRTY = ":௩௦"  # :30
TA_TIME_FORTYFIVE = ":௪௫"  # :45

# Arabic time patterns
AR_TIME_FIFTEEN = ":15"
AR_TIME_THIRTY = ":30"
AR_TIME_FORTYFIVE = ":45"

# Convert Arabic digits (0-9) to Tamil digits (௦-௯)
arabic_to_tamil_digit = pynini.string_map([
    ("0", "௦"), ("1", "௧"), ("2", "௨"), ("3", "௩"), ("4", "௪"),
    ("5", "௫"), ("6", "௬"), ("7", "௭"), ("8", "௮"), ("9", "௯")
]).optimize()
arabic_to_tamil_number = pynini.closure(arabic_to_tamil_digit).optimize()

hours_graph = pynini.string_file(get_abs_path("data/time/hours.tsv"))
minutes_graph = pynini.string_file(get_abs_path("data/time/minutes.tsv"))
seconds_graph = pynini.string_file(get_abs_path("data/time/seconds.tsv"))


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        ௧௨:௩௦:௩௦  -> time { hours: "பன்னிரண்டு" minutes: "முப்பது" seconds: "முப்பது" }
        ௧:௪௦  -> time { hours: "ஒன்று" minutes: "நாற்பது" }
        ௧:௦௦  -> time { hours: "ஒன்று" }

    Args:
        time: GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="time", kind="classify")

        delete_colon = pynutil.delete(":")
        cardinal_graph = cardinal.digit | cardinal.teens_and_ties

        # Delete optional leading zero (handles inputs like 09, 07, 00)
        delete_leading_zero_tamil = (
            (NEMO_TA_NON_ZERO + NEMO_TA_DIGIT)  # keep 10-99 as-is
            | (pynutil.delete(NEMO_TA_ZERO) + NEMO_TA_DIGIT)  # drop leading zero for 00-09
            | NEMO_TA_DIGIT  # allow single-digit inputs
        ).optimize()
        delete_leading_zero_arabic = (
            (pynini.difference(NEMO_DIGIT, "0") + NEMO_DIGIT)  # keep 10-99 as-is
            | (pynutil.delete("0") + NEMO_DIGIT)  # drop leading zero for 00-09
            | NEMO_DIGIT  # allow single-digit inputs
        ).optimize()

        # Support both Tamil and Arabic digits for hours and minutes
        # Create combined graphs that accept both Arabic and Tamil digits
        # Tamil digits path: delete optional leading zero -> hours_graph
        tamil_hour_path = pynini.compose(delete_leading_zero_tamil, hours_graph).optimize()
        # Arabic digits path: delete optional leading zero -> convert to Tamil -> hours_graph
        arabic_hour_path = pynini.compose(
            delete_leading_zero_arabic,
            arabic_to_tamil_number @ hours_graph
        ).optimize()
        hour_input = tamil_hour_path | arabic_hour_path

        tamil_minute_path = pynini.compose(pynini.closure(NEMO_TA_DIGIT, 1), minutes_graph).optimize()
        arabic_minute_path = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_tamil_number @ minutes_graph
        ).optimize()
        minute_input = tamil_minute_path | arabic_minute_path

        tamil_second_path = pynini.compose(pynini.closure(NEMO_TA_DIGIT, 1), seconds_graph).optimize()
        arabic_second_path = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_tamil_number @ seconds_graph
        ).optimize()
        second_input = tamil_second_path | arabic_second_path

        self.hours = pynutil.insert("hours: \"") + hour_input + pynutil.insert("\" ")
        self.minutes = pynutil.insert("minutes: \"") + minute_input + pynutil.insert("\" ")
        self.seconds = pynutil.insert("seconds: \"") + second_input + pynutil.insert("\" ")

        # Optional "மணிக்கு" after time (to avoid duplication when verbalizer adds it)
        # Handle optional space(s) before "மணிக்கு"
        optional_manikku = pynini.closure(
            pynini.closure(NEMO_SPACE, 0, 1) + pynutil.delete("மணிக்கு"), 0, 1
        ).optimize()

        # hour minute seconds
        graph_hms = (
            self.hours + delete_colon + insert_space + self.minutes + delete_colon + insert_space + self.seconds + optional_manikku
        )

        # hour minute - NORMAL FORMAT (highest priority)
        graph_hm = self.hours + delete_colon + insert_space + self.minutes + optional_manikku

        # hour
        arabic_double_zero = pynutil.delete("00")
        graph_h = (
            self.hours
            + delete_colon
            + (pynutil.delete(TA_DOUBLE_ZERO) | arabic_double_zero)
            + optional_manikku
        )

        dedh_dhai_graph = (
            pynini.string_map([("௧" + TA_TIME_THIRTY, TA_DEDH), ("௨" + TA_TIME_THIRTY, TA_DHAI)])
            | pynini.string_map([("1" + AR_TIME_THIRTY, TA_DEDH), ("2" + AR_TIME_THIRTY, TA_DHAI)])
        )

        savva_numbers = (
            cardinal_graph + pynini.cross(TA_TIME_FIFTEEN, "")
            | cardinal_graph + pynini.cross(AR_TIME_FIFTEEN, "")
        )
        savva_graph = pynutil.insert(TA_SAVVA) + pynutil.insert(NEMO_SPACE) + savva_numbers

        sadhe_numbers = (
            cardinal_graph + pynini.cross(TA_TIME_THIRTY, "")
            | cardinal_graph + pynini.cross(AR_TIME_THIRTY, "")
        )
        sadhe_graph = pynutil.insert(TA_SADHE) + pynutil.insert(NEMO_SPACE) + sadhe_numbers

        paune = pynini.string_file(get_abs_path("data/whitelist/paune_mappings.tsv"))
        paune_numbers = (
            paune + pynini.cross(TA_TIME_FORTYFIVE, "")
            | (arabic_to_tamil_number @ paune) + pynini.cross(AR_TIME_FORTYFIVE, "")
        )
        paune_graph = pynutil.insert(TA_PAUNE) + pynutil.insert(NEMO_SPACE) + paune_numbers

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

        # Prioritize normal hour:minute format over special Tamil time expressions
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

