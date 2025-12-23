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

from indic_text_normalization.bn.graph_utils import (
    NEMO_DIGIT,
    NEMO_BN_DIGIT,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from indic_text_normalization.bn.utils import get_abs_path

# Convert Arabic digits (0-9) to Bengali digits (০-৯)
arabic_to_bengali_digit = pynini.string_map([
    ("0", "০"), ("1", "১"), ("2", "২"), ("3", "৩"), ("4", "৪"),
    ("5", "৫"), ("6", "৬"), ("7", "৭"), ("8", "৮"), ("9", "৯")
]).optimize()
arabic_to_bengali_number = pynini.closure(arabic_to_bengali_digit).optimize()

# Pad single digit to two digits (prepend 0)
pad_single_digit_bengali = (
    (pynutil.insert("০") + NEMO_BN_DIGIT)  # single digit -> prepend ০
).optimize()
pad_single_digit_arabic = (
    (pynutil.insert("0") + NEMO_DIGIT)  # single digit -> prepend 0
).optimize()

# Bengali time patterns
BN_DOUBLE_ZERO = "০০"

hours_graph = pynini.string_file(get_abs_path("data/time/hours.tsv"))
minutes_graph = pynini.string_file(get_abs_path("data/time/minutes.tsv"))
seconds_graph = pynini.string_file(get_abs_path("data/time/seconds.tsv"))


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        ১২:৩০:৩০  -> time { hours: "বারো" minutes: "ত্রিশ" seconds: "ত্রিশ" }
        ১:৪০  -> time { hours: "এক" minutes: "চল্লিশ" }
        ১:০০  -> time { hours: "এক" }
        12:30  -> time { hours: "বারো" minutes: "ত্রিশ" }
        09:05  -> time { hours: "নয়" minutes: "পাঁচ" }

    Args:
        time: GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="time", kind="classify")

        delete_colon = pynutil.delete(":")

        # Hours: accept 1 or 2 Bengali digits, pad single digit with leading zero
        bengali_hour_2digit = pynini.closure(NEMO_BN_DIGIT, 2, 2)  # 2 digits: use as-is
        bengali_hour_1digit = pad_single_digit_bengali  # 1 digit: prepend ০
        bengali_hour_input = bengali_hour_2digit | bengali_hour_1digit
        bengali_hour_path = pynini.compose(bengali_hour_input, hours_graph).optimize()

        # Hours: accept 1 or 2 Arabic digits, pad single digit, then convert to Bengali
        arabic_hour_2digit = pynini.closure(NEMO_DIGIT, 2, 2)  # 2 digits: use as-is
        arabic_hour_1digit = pad_single_digit_arabic  # 1 digit: prepend 0
        arabic_hour_input = arabic_hour_2digit | arabic_hour_1digit
        arabic_hour_path = pynini.compose(
            arabic_hour_input,
            arabic_to_bengali_number @ hours_graph
        ).optimize()
        
        hour_input = bengali_hour_path | arabic_hour_path

        # Minutes: accept 1 or 2 Bengali digits, pad single digit with leading zero
        bengali_minute_2digit = pynini.closure(NEMO_BN_DIGIT, 2, 2)
        bengali_minute_1digit = pad_single_digit_bengali
        bengali_minute_input = bengali_minute_2digit | bengali_minute_1digit
        bengali_minute_path = pynini.compose(bengali_minute_input, minutes_graph).optimize()

        arabic_minute_2digit = pynini.closure(NEMO_DIGIT, 2, 2)
        arabic_minute_1digit = pad_single_digit_arabic
        arabic_minute_input = arabic_minute_2digit | arabic_minute_1digit
        arabic_minute_path = pynini.compose(
            arabic_minute_input,
            arabic_to_bengali_number @ minutes_graph
        ).optimize()
        
        minute_input = bengali_minute_path | arabic_minute_path

        # Seconds: accept 1 or 2 Bengali digits, pad single digit with leading zero
        bengali_second_2digit = pynini.closure(NEMO_BN_DIGIT, 2, 2)
        bengali_second_1digit = pad_single_digit_bengali
        bengali_second_input = bengali_second_2digit | bengali_second_1digit
        bengali_second_path = pynini.compose(bengali_second_input, seconds_graph).optimize()

        arabic_second_2digit = pynini.closure(NEMO_DIGIT, 2, 2)
        arabic_second_1digit = pad_single_digit_arabic
        arabic_second_input = arabic_second_2digit | arabic_second_1digit
        arabic_second_path = pynini.compose(
            arabic_second_input,
            arabic_to_bengali_number @ seconds_graph
        ).optimize()
        
        second_input = bengali_second_path | arabic_second_path

        self.hours = pynutil.insert("hours: \"") + hour_input + pynutil.insert("\" ")
        self.minutes = pynutil.insert("minutes: \"") + minute_input + pynutil.insert("\" ")
        self.seconds = pynutil.insert("seconds: \"") + second_input + pynutil.insert("\" ")

        # Optional "টায়" after time (to avoid duplication when verbalizer adds it)
        optional_tay = pynini.closure(
            pynini.closure(NEMO_SPACE, 0, 1) + pynutil.delete("টায়"), 0, 1
        ).optimize()

        # hour minute seconds
        graph_hms = (
            self.hours + delete_colon + insert_space + self.minutes + delete_colon + insert_space + self.seconds + optional_tay
        )

        # hour minute - NORMAL FORMAT
        graph_hm = self.hours + delete_colon + insert_space + self.minutes + optional_tay

        # hour only (HH:00) - support both Bengali and Arabic double zero
        arabic_double_zero = pynutil.delete("00")
        graph_h = (
            self.hours
            + delete_colon
            + (pynutil.delete(BN_DOUBLE_ZERO) | arabic_double_zero)
            + optional_tay
        )

        # Simple weight ordering: HMS first, then HM, then H only
        # Lower weight = higher priority in pynini
        final_graph = (
            pynutil.add_weight(graph_hms, -0.1)  # Highest priority
            | pynutil.add_weight(graph_hm, -0.05)  # Second priority
            | graph_h  # Default weight (0)
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

