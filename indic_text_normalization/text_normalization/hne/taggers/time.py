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

from indic_text_normalization.text_normalization.cg.graph_utils import (
    CG_DEDH,
    CG_DHAI,
    CG_PAUNE,
    CG_SADHE,
    CG_SAVVA,
    NEMO_DIGIT,
    NEMO_CG_DIGIT,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from indic_text_normalization.text_normalization.cg.utils import get_abs_path

# Time patterns specific to time tagger
CG_DOUBLE_ZERO = "००"
CG_TIME_FIFTEEN = ":१५"  # :15
CG_TIME_THIRTY = ":३०"  # :30
CG_TIME_FORTYFIVE = ":४५"  # :45

# Arabic time patterns
AR_TIME_FIFTEEN = ":15"
AR_TIME_THIRTY = ":30"
AR_TIME_FORTYFIVE = ":45"

# Convert Arabic digits (0-9) to Chhattisgarhi digits (०-९)
arabic_to_cg_digit = pynini.string_map([
    ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
    ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९")
]).optimize()
arabic_to_cg_number = pynini.closure(arabic_to_cg_digit).optimize()

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

        # Support both Chhattisgarhi and Arabic digits for hours and minutes
        # Create combined graphs that accept both Arabic and Chhattisgarhi digits
        # Chhattisgarhi digits path: Chhattisgarhi digits -> hours_graph
        chhattisgarhi_hour_path = pynini.compose(pynini.closure(NEMO_CG_DIGIT, 1), hours_graph).optimize()
        # Arabic digits path: Arabic digits -> convert to Chhattisgarhi -> hours_graph
        arabic_hour_path = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1), 
            arabic_to_cg_number @ hours_graph
        ).optimize()
        hour_input = chhattisgarhi_hour_path | arabic_hour_path

        chhattisgarhi_minute_path = pynini.compose(pynini.closure(NEMO_CG_DIGIT, 1), minutes_graph).optimize()
        arabic_minute_path = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_cg_number @ minutes_graph
        ).optimize()
        minute_input = chhattisgarhi_minute_path | arabic_minute_path

        chhattisgarhi_second_path = pynini.compose(pynini.closure(NEMO_CG_DIGIT, 1), seconds_graph).optimize()
        arabic_second_path = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_cg_number @ seconds_graph
        ).optimize()
        second_input = chhattisgarhi_second_path | arabic_second_path

        self.hours = pynutil.insert("hours: \"") + hour_input + pynutil.insert("\" ")
        self.minutes = pynutil.insert("minutes: \"") + minute_input + pynutil.insert("\" ")
        self.seconds = pynutil.insert("seconds: \"") + second_input + pynutil.insert("\" ")

        # hour minute seconds
        graph_hms = (
            self.hours + delete_colon + insert_space + self.minutes + delete_colon + insert_space + self.seconds
        )

        # hour minute
        graph_hm = self.hours + delete_colon + insert_space + self.minutes

        # hour - support both Chhattisgarhi and Arabic double zero
        chhattisgarhi_double_zero = pynutil.delete(CG_DOUBLE_ZERO)
        arabic_double_zero = pynutil.delete("00")
        double_zero = chhattisgarhi_double_zero | arabic_double_zero
        graph_h = self.hours + delete_colon + double_zero

        # Support both Chhattisgarhi and Arabic time patterns for dedh/dhai
        dedh_dhai_graph = (
            pynini.string_map([("१" + CG_TIME_THIRTY, CG_DEDH), ("२" + CG_TIME_THIRTY, CG_DHAI)])
            | pynini.string_map([("1" + AR_TIME_THIRTY, CG_DEDH), ("2" + AR_TIME_THIRTY, CG_DHAI)])
        )

        # Support both Chhattisgarhi and Arabic time patterns
        savva_numbers = (
            (cardinal_graph + pynini.cross(CG_TIME_FIFTEEN, ""))
            | (cardinal_graph + pynini.cross(AR_TIME_FIFTEEN, ""))
        )
        savva_graph = pynutil.insert(CG_SAVVA) + pynutil.insert(NEMO_SPACE) + savva_numbers

        sadhe_numbers = (
            (cardinal_graph + pynini.cross(CG_TIME_THIRTY, ""))
            | (cardinal_graph + pynini.cross(AR_TIME_THIRTY, ""))
        )
        sadhe_graph = pynutil.insert(CG_SADHE) + pynutil.insert(NEMO_SPACE) + sadhe_numbers

        paune = pynini.string_file(get_abs_path("data/whitelist/paune_mappings.tsv"))
        paune_numbers = (
            (paune + pynini.cross(CG_TIME_FORTYFIVE, ""))
            | (paune + pynini.cross(AR_TIME_FORTYFIVE, ""))
        )
        paune_graph = pynutil.insert(CG_PAUNE) + pynutil.insert(NEMO_SPACE) + paune_numbers

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
