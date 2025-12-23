# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
    NEMO_BN_DIGIT,
    NEMO_DIGIT,
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

days = pynini.string_file(get_abs_path("data/date/days.tsv"))
months = pynini.string_file(get_abs_path("data/date/months.tsv"))
year_suffix = pynini.string_file(get_abs_path("data/date/year_suffix.tsv"))


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, e.g.
        "০১-০৪-২০২৪" -> date { day: "এক" month: "এপ্রিল" year: "দুই হাজার চব্বিশ" }
        "01-04-2024" -> date { day: "এক" month: "এপ্রিল" year: "দুই হাজার চব্বিশ" }
        "2024-01-15" -> date { year: "দুই হাজার চব্বিশ" month: "জানুয়ারি" day: "পনেরো" }

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="date", kind="classify")

        cardinal_graph = cardinal.final_graph

        # Support both Bengali and Arabic digits for days
        # Bengali digits path: Bengali digits -> days mapping
        bengali_day_input = pynini.closure(NEMO_BN_DIGIT, 1, 2)
        bengali_days_graph = pynini.compose(bengali_day_input, days).optimize()
        
        # Arabic digits path: Arabic digits -> convert to Bengali -> days mapping
        arabic_day_input = pynini.closure(NEMO_DIGIT, 1, 2)
        arabic_days_graph = pynini.compose(
            arabic_day_input,
            arabic_to_bengali_number @ days
        ).optimize()
        
        days_graph = bengali_days_graph | arabic_days_graph

        # Support both Bengali and Arabic digits for months
        bengali_month_input = pynini.closure(NEMO_BN_DIGIT, 1, 2)
        bengali_months_graph = pynini.compose(bengali_month_input, months).optimize()
        
        arabic_month_input = pynini.closure(NEMO_DIGIT, 1, 2)
        arabic_months_graph = pynini.compose(
            arabic_month_input,
            arabic_to_bengali_number @ months
        ).optimize()
        
        months_graph = bengali_months_graph | arabic_months_graph

        # Year graph - support both Bengali and Arabic 4-digit years
        bengali_year_input = NEMO_BN_DIGIT + NEMO_BN_DIGIT + NEMO_BN_DIGIT + NEMO_BN_DIGIT
        bengali_year_graph = pynini.compose(bengali_year_input, cardinal_graph).optimize()
        
        arabic_year_input = NEMO_DIGIT + NEMO_DIGIT + NEMO_DIGIT + NEMO_DIGIT
        arabic_year_graph = pynini.compose(
            arabic_year_input,
            arabic_to_bengali_number @ cardinal_graph
        ).optimize()
        
        year_graph = bengali_year_graph | arabic_year_graph

        # Also support 2-digit years
        bengali_year_2digit_input = NEMO_BN_DIGIT + NEMO_BN_DIGIT
        bengali_year_2digit_graph = pynini.compose(bengali_year_2digit_input, cardinal_graph).optimize()
        
        arabic_year_2digit_input = NEMO_DIGIT + NEMO_DIGIT
        arabic_year_2digit_graph = pynini.compose(
            arabic_year_2digit_input,
            arabic_to_bengali_number @ cardinal_graph
        ).optimize()
        
        year_2digit_graph = bengali_year_2digit_graph | arabic_year_2digit_graph

        # Separators
        delete_dash = pynutil.delete("-")
        delete_slash = pynutil.delete("/")
        delete_dot = pynutil.delete(".")
        delete_separator = delete_dash | delete_slash | delete_dot

        # Build date components with labels
        day_component = pynutil.insert("day: \"") + days_graph + pynutil.insert("\"")
        month_component = pynutil.insert("month: \"") + months_graph + pynutil.insert("\"")
        year_component = pynutil.insert("year: \"") + (year_graph | year_2digit_graph) + pynutil.insert("\"")

        # DD-MM-YYYY format (common in India)
        graph_dd_mm_yyyy = (
            day_component
            + insert_space
            + delete_separator
            + month_component
            + insert_space
            + delete_separator
            + year_component
        )

        # MM-DD-YYYY format
        graph_mm_dd_yyyy = (
            month_component
            + insert_space
            + delete_separator
            + day_component
            + insert_space
            + delete_separator
            + year_component
            + pynutil.insert(" preserve_order: true")
        )

        # YYYY-MM-DD format (ISO format)
        graph_yyyy_mm_dd = (
            year_component
            + insert_space
            + delete_separator
            + month_component
            + insert_space
            + delete_separator
            + day_component
        )

        # DD-MM format (without year)
        graph_dd_mm = (
            day_component
            + insert_space
            + delete_separator
            + month_component
        )

        # MM-DD format (without year)
        graph_mm_dd = (
            month_component
            + insert_space
            + delete_separator
            + day_component
            + pynutil.insert(" preserve_order: true")
        )

        # Year suffix (খ্রিস্টাব্দ, etc.)
        era_graph = pynutil.insert("era: \"") + year_suffix + pynutil.insert("\"")

        # Combine all date formats with weights
        final_graph = (
            pynutil.add_weight(graph_dd_mm_yyyy, -0.001)  # Prefer DD-MM-YYYY
            | pynutil.add_weight(graph_yyyy_mm_dd, -0.001)  # ISO format
            | graph_mm_dd_yyyy
            | pynutil.add_weight(graph_dd_mm, -0.002)
            | graph_mm_dd
            | pynutil.add_weight(era_graph, -0.001)
        )

        self.final_graph = final_graph.optimize()
        self.fst = self.add_tokens(self.final_graph)

