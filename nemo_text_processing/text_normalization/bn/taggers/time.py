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

from nemo_text_processing.text_normalization.bn.graph_utils import (
    NEMO_DIGIT,
    NEMO_BN_DIGIT,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.bn.utils import get_abs_path

# Convert Arabic digits (0-9) to Bengali digits (০-৯)
arabic_to_bengali_digit = pynini.string_map([
    ("0", "০"), ("1", "১"), ("2", "২"), ("3", "৩"), ("4", "৪"),
    ("5", "৫"), ("6", "৬"), ("7", "৭"), ("8", "৮"), ("9", "৯")
]).optimize()
arabic_to_bengali_number = pynini.closure(arabic_to_bengali_digit).optimize()

hours_graph = pynini.string_file(get_abs_path("data/time/hours.tsv"))
minutes_graph = pynini.string_file(get_abs_path("data/time/minutes.tsv"))
seconds_graph = pynini.string_file(get_abs_path("data/time/seconds.tsv"))


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        ১২:৩০:৩০  -> time { hours: "বারো" minutes: "ত্রিশ" seconds: "ত্রিশ" }
        ১:৪০  -> time { hours: "এক" minutes: "চল্লিশ" }
        ১:০০  -> time { hours: "এক" }

    Args:
        time: GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="time", kind="classify")

        delete_colon = pynutil.delete(":")
        cardinal_graph = cardinal.digit | cardinal.teens_and_ties

        # Support both Bengali and Arabic digits for hours (1-2 digits: 0-23)
        bengali_hour_path = pynini.compose(pynini.closure(NEMO_BN_DIGIT, 1, 2), hours_graph).optimize()
        arabic_hour_path = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1, 2), 
            arabic_to_bengali_number @ hours_graph
        ).optimize()
        hour_input = bengali_hour_path | arabic_hour_path

        # Minutes: support both 1 and 2 digits (0-59)
        # For 2 digits, use minutes_graph directly
        bengali_minute_two = pynini.compose(
            pynini.closure(NEMO_BN_DIGIT, 2, 2), 
            minutes_graph
        ).optimize()
        # For 1 digit, convert to Bengali and use cardinal
        bengali_minute_one = pynini.compose(
            pynini.closure(NEMO_BN_DIGIT, 1, 1),
            cardinal_graph
        ).optimize()
        bengali_minute_path = bengali_minute_two | bengali_minute_one
        
        # Create a converter for exactly 2 digits
        arabic_to_bengali_two_digits = (
            arabic_to_bengali_digit + arabic_to_bengali_digit
        ).optimize()
        arabic_minute_two = pynini.compose(
            pynini.closure(NEMO_DIGIT, 2, 2),
            arabic_to_bengali_two_digits @ minutes_graph
        ).optimize()
        arabic_minute_one = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1, 1),
            arabic_to_bengali_number @ cardinal_graph
        ).optimize()
        arabic_minute_path = arabic_minute_two | arabic_minute_one
        minute_input = bengali_minute_path | arabic_minute_path

        # Seconds: support both 1 and 2 digits (0-59)
        # For 2 digits, use seconds_graph directly
        bengali_second_two = pynini.compose(
            pynini.closure(NEMO_BN_DIGIT, 2, 2), 
            seconds_graph
        ).optimize()
        # For 1 digit, convert to Bengali and use cardinal
        bengali_second_one = pynini.compose(
            pynini.closure(NEMO_BN_DIGIT, 1, 1),
            cardinal_graph
        ).optimize()
        bengali_second_path = bengali_second_two | bengali_second_one
        
        arabic_second_two = pynini.compose(
            pynini.closure(NEMO_DIGIT, 2, 2),
            arabic_to_bengali_two_digits @ seconds_graph
        ).optimize()
        arabic_second_one = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1, 1),
            arabic_to_bengali_number @ cardinal_graph
        ).optimize()
        arabic_second_path = arabic_second_two | arabic_second_one
        second_input = bengali_second_path | arabic_second_path

        self.hours = pynutil.insert("hours: \"") + hour_input + pynutil.insert("\" ")
        self.minutes = pynutil.insert("minutes: \"") + minute_input + pynutil.insert("\" ")
        self.seconds = pynutil.insert("seconds: \"") + second_input + pynutil.insert("\" ")

        # hour minute seconds
        graph_hms = (
            self.hours + delete_colon + insert_space + self.minutes + delete_colon + insert_space + self.seconds
        )

        # hour minute
        graph_hm = self.hours + delete_colon + insert_space + self.minutes

        # hour - support both Bengali and Arabic double zero
        bengali_double_zero = pynutil.delete("০০")
        arabic_double_zero = pynutil.delete("00")
        double_zero = bengali_double_zero | arabic_double_zero
        graph_h = self.hours + delete_colon + double_zero

        final_graph = (
            graph_hms
            | pynutil.add_weight(graph_hm, 0.3)
            | pynutil.add_weight(graph_h, 0.3)
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

