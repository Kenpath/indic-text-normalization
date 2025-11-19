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

from nemo_text_processing.text_normalization.gu.graph_utils import (
    NEMO_GU_ZERO,
    NEMO_DIGIT,
    NEMO_GU_DIGIT,
    GU_DEDH,
    GU_DHAI,
    GU_PAUNE,
    GU_SADHE,
    GU_SAVVA,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.gu.utils import get_abs_path

# Time patterns specific to time tagger
GU_DOUBLE_ZERO = "૦૦"
GU_TIME_FIFTEEN = ":૧૫"  # :15
GU_TIME_THIRTY = ":૩૦"  # :30
GU_TIME_FORTYFIVE = ":૪૫"  # :45

# Arabic time patterns
AR_TIME_FIFTEEN = ":15"
AR_TIME_THIRTY = ":30"
AR_TIME_FORTYFIVE = ":45"

# Convert Arabic digits (0-9) to Gujarati digits (૦-૯)
arabic_to_gujarati_digit = pynini.string_map([
    ("0", "૦"), ("1", "૧"), ("2", "૨"), ("3", "૩"), ("4", "૪"),
    ("5", "૫"), ("6", "૬"), ("7", "૭"), ("8", "૮"), ("9", "૯")
]).optimize()
arabic_to_gujarati_number = pynini.closure(arabic_to_gujarati_digit).optimize()

hours_graph = pynini.string_file(get_abs_path("data/time/hours.tsv"))
minutes_graph = pynini.string_file(get_abs_path("data/time/minutes.tsv"))
seconds_graph = pynini.string_file(get_abs_path("data/time/seconds.tsv"))


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        ૧૨:૩૦:૩૦  -> time { hours: "બાર" minutes: "ત્રીસ" seconds: "ત્રીસ" }
        ૧:૪૦  -> time { hours: "એક" minutes: "ચાલીસ" }
        ૧:૦૦  -> time { hours: "એક" }

    Args:
        time: GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="time", kind="classify")

        delete_colon = pynutil.delete(":")
        cardinal_graph = cardinal.digit | cardinal.teens_and_ties

        # Support both Gujarati and Arabic digits for hours and minutes
        # Create combined graphs that accept both Arabic and Gujarati digits
        # Gujarati digits path: Gujarati digits -> hours_graph
        gujarati_hour_path = pynini.compose(pynini.closure(NEMO_GU_DIGIT, 1), hours_graph).optimize()
        # Arabic digits path: Arabic digits -> convert to Gujarati -> hours_graph
        arabic_hour_path = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1), 
            arabic_to_gujarati_number @ hours_graph
        ).optimize()
        hour_input = gujarati_hour_path | arabic_hour_path

        gujarati_minute_path = pynini.compose(pynini.closure(NEMO_GU_DIGIT, 1), minutes_graph).optimize()
        arabic_minute_path = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_gujarati_number @ minutes_graph
        ).optimize()
        minute_input = gujarati_minute_path | arabic_minute_path

        gujarati_second_path = pynini.compose(pynini.closure(NEMO_GU_DIGIT, 1), seconds_graph).optimize()
        arabic_second_path = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_gujarati_number @ seconds_graph
        ).optimize()
        second_input = gujarati_second_path | arabic_second_path

        self.hours = pynutil.insert("hours: \"") + hour_input + pynutil.insert("\" ")
        self.minutes = pynutil.insert("minutes: \"") + minute_input + pynutil.insert("\" ")
        self.seconds = pynutil.insert("seconds: \"") + second_input + pynutil.insert("\" ")

        # Optional "વાગ્યે" after time (to avoid duplication when verbalizer adds it)
        # Handle optional space(s) before "વાગ્યે"
        optional_vagye = pynini.closure(
            pynini.closure(NEMO_SPACE, 0, 1) + pynutil.delete("વાગ્યે"), 0, 1
        ).optimize()

        # hour minute seconds
        graph_hms = (
            self.hours + delete_colon + insert_space + self.minutes + delete_colon + insert_space + self.seconds + optional_vagye
        )

        # hour minute - NORMAL FORMAT (highest priority)
        graph_hm = self.hours + delete_colon + insert_space + self.minutes + optional_vagye

        # hour
        graph_h = self.hours + delete_colon + pynutil.delete(GU_DOUBLE_ZERO) + optional_vagye

        dedh_dhai_graph = pynini.string_map([("૧" + GU_TIME_THIRTY, GU_DEDH), ("૨" + GU_TIME_THIRTY, GU_DHAI)])

        savva_numbers = cardinal_graph + pynini.cross(GU_TIME_FIFTEEN, "")
        savva_graph = pynutil.insert(GU_SAVVA) + pynutil.insert(NEMO_SPACE) + savva_numbers

        sadhe_numbers = cardinal_graph + pynini.cross(GU_TIME_THIRTY, "")
        sadhe_graph = pynutil.insert(GU_SADHE) + pynutil.insert(NEMO_SPACE) + sadhe_numbers

        paune = pynini.string_file(get_abs_path("data/whitelist/paune_mappings.tsv"))
        paune_numbers = paune + pynini.cross(GU_TIME_FORTYFIVE, "")
        paune_graph = pynutil.insert(GU_PAUNE) + pynutil.insert(NEMO_SPACE) + paune_numbers

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

        # Prioritize normal hour:minute format over special Gujarati time expressions
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

