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

from ..graph_utils import (
    NEMO_DIGIT,
    NEMO_HI_DIGIT,
    NEMO_HI_NON_ZERO,
    NEMO_HI_ZERO,
    GraphFst,
    insert_space,
)
from ..utils import get_abs_path

# Convert Arabic digits (0-9) to Hindi digits (०-९)
arabic_to_hindi_digit = pynini.string_map([
    ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
    ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९")
]).optimize()
arabic_to_hindi_number = pynini.closure(arabic_to_hindi_digit).optimize()

days = pynini.string_file(get_abs_path("data/date/days.tsv"))
months = pynini.string_file(get_abs_path("data/date/months.tsv"))
year_suffix = pynini.string_file(get_abs_path("data/date/year_suffix.tsv"))
digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
teens_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
teens_and_ties = pynutil.add_weight(teens_ties, -0.1)

# Read suffixes from file into a list
with open(get_abs_path("data/date/suffixes.tsv"), "r", encoding="utf-8") as f:
    suffixes_list = f.read().splitlines()
with open(get_abs_path("data/date/prefixes.tsv"), "r", encoding="utf-8") as f:
    prefixes_list = f.read().splitlines()

# Create union of suffixes and prefixes
suffix_union = pynini.union(*suffixes_list)
prefix_union = pynini.union(*prefixes_list)


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, e.g.
        "०१-०४-२०२४" -> date { day: "एक" month: "अप्रैल" year: "दो हज़ार चौबीस" }
        "०४-०१-२०२४" -> date { month: "अप्रैल" day: "एक" year: "दो हज़ार चौबीस" }


    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="date", kind="classify")

        # Support both Hindi and Arabic digits for dates
        # Year patterns: 4-digit years (e.g., 2024, २०२४)
        # Pattern for thousands: X0XX (e.g., 2024 -> 2०24)
        # Pattern for hundreds: X1-9XX (e.g., 1999 -> 1९99)
        
        # Year pattern definitions
        year_pattern_thousands = (NEMO_HI_DIGIT + NEMO_HI_ZERO + NEMO_HI_DIGIT + NEMO_HI_DIGIT)
        year_pattern_hundreds = (NEMO_HI_DIGIT + NEMO_HI_NON_ZERO + NEMO_HI_DIGIT + NEMO_HI_DIGIT)
        
        # Hindi digits for year patterns
        hindi_year_thousands = pynini.compose(
            year_pattern_thousands, cardinal.graph_thousands
        )
        hindi_year_hundreds_as_thousands = pynini.compose(
            year_pattern_hundreds, cardinal.graph_hundreds_as_thousand
        )
        
        # Arabic digits for year patterns - convert to Hindi first
        # Convert 4-digit Arabic year (e.g., "2024") to Hindi ("२०२४"), then match patterns
        arabic_year_4digits = (NEMO_DIGIT + NEMO_DIGIT + NEMO_DIGIT + NEMO_DIGIT)
        # Convert Arabic to Hindi
        arabic_to_hindi_year = arabic_year_4digits @ arabic_to_hindi_number
        
        # Match converted Hindi year against patterns and compose with cardinal
        arabic_year_thousands = pynini.compose(
            arabic_to_hindi_year,
            pynini.compose(year_pattern_thousands, cardinal.graph_thousands)
        )
        arabic_year_hundreds_as_thousands = pynini.compose(
            arabic_to_hindi_year,
            pynini.compose(year_pattern_hundreds, cardinal.graph_hundreds_as_thousand)
        )
        
        # Combined year graphs (supports both Hindi and Arabic digits)
        graph_year_thousands = hindi_year_thousands | arabic_year_thousands
        graph_year_hundreds_as_thousands = hindi_year_hundreds_as_thousands | arabic_year_hundreds_as_thousands

        cardinal_graph = pynini.union(
            digit, teens_and_ties, cardinal.graph_hundreds, graph_year_thousands, graph_year_hundreds_as_thousands
        )

        graph_year = pynini.union(graph_year_thousands, graph_year_hundreds_as_thousands)

        delete_dash = pynutil.delete("-")
        delete_slash = pynutil.delete("/")

        # Support both Hindi and Arabic digits for days and months
        # Hindi digits path: Hindi digits -> days/months mapping
        hindi_days_graph = pynutil.insert("day: \"") + days + pynutil.insert("\"") + insert_space
        hindi_months_graph = pynutil.insert("month: \"") + months + pynutil.insert("\"") + insert_space
        
        # Arabic digits path: Arabic digits -> convert to Hindi -> days/months mapping
        # Day pattern: 1-31 (can have leading zero: 01-09, or no leading zero: 1-31)
        # Match 1-2 digit Arabic numbers
        arabic_day_input = pynini.closure(NEMO_DIGIT, 1, 2)
        arabic_days_graph = pynutil.insert("day: \"") + pynini.compose(
            arabic_day_input,
            arabic_to_hindi_number @ days
        ) + pynutil.insert("\"") + insert_space
        
        # Month pattern: 1-12 (can have leading zero: 01-09, or no leading zero: 1-12)
        # Match 1-2 digit Arabic numbers
        arabic_month_input = pynini.closure(NEMO_DIGIT, 1, 2)
        arabic_months_graph = pynutil.insert("month: \"") + pynini.compose(
            arabic_month_input,
            arabic_to_hindi_number @ months
        ) + pynutil.insert("\"") + insert_space
        
        # Combined graphs (supports both Hindi and Arabic digits)
        days_graph = hindi_days_graph | arabic_days_graph
        months_graph = hindi_months_graph | arabic_months_graph

        years_graph = pynutil.insert("year: \"") + graph_year + pynutil.insert("\"") + insert_space

        graph_dd_mm = days_graph + delete_dash + months_graph

        graph_mm_dd = months_graph + delete_dash + days_graph

        graph_mm_dd += pynutil.insert(" preserve_order: true ")

        # Graph for era
        era_graph = pynutil.insert("era: \"") + year_suffix + pynutil.insert("\"") + insert_space

        range_graph = pynini.cross("-", "से")

        # Graph for year - support both Hindi and Arabic digits
        # Hindi digits path
        hindi_century_input = pynini.closure(NEMO_HI_DIGIT, 1)
        hindi_century_number = pynini.compose(hindi_century_input, cardinal_graph) + pynini.accep("वीं")
        
        # Arabic digits path
        arabic_century_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_century_number = pynini.compose(
            arabic_century_input,
            arabic_to_hindi_number @ cardinal_graph
        ) + pynini.accep("वीं")
        
        century_number = hindi_century_number | arabic_century_number
        century_text = pynutil.insert("era: \"") + century_number + pynutil.insert("\"") + insert_space

        # Updated logic to use suffix_union
        year_number = graph_year + suffix_union
        year_text = pynutil.insert("era: \"") + year_number + pynutil.insert("\"") + insert_space

        # Updated logic to use prefix_union
        year_prefix = pynutil.insert("era: \"") + prefix_union + insert_space + graph_year + pynutil.insert("\"")

        delete_separator = pynini.union(delete_dash, delete_slash)
        graph_dd_mm_yyyy = days_graph + delete_separator + months_graph + delete_separator + years_graph

        graph_mm_dd_yyyy = months_graph + delete_separator + days_graph + delete_separator + years_graph

        graph_mm_dd_yyyy += pynutil.insert(" preserve_order: true ")

        graph_mm_yyyy = months_graph + delete_dash + insert_space + years_graph

        graph_year_suffix = era_graph

        graph_range = (
            pynutil.insert("era: \"")
            + cardinal_graph
            + insert_space
            + range_graph
            + insert_space
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(" preserve_order: true ")
        )

        # default assume dd_mm_yyyy

        final_graph = (
            pynutil.add_weight(graph_dd_mm, -0.001)
            | graph_mm_dd
            | pynutil.add_weight(graph_dd_mm_yyyy, -0.001)
            | graph_mm_dd_yyyy
            | pynutil.add_weight(graph_mm_yyyy, -0.2)
            | pynutil.add_weight(graph_year_suffix, -0.001)
            | pynutil.add_weight(graph_range, -0.005)
            | pynutil.add_weight(century_text, -0.001)
            | pynutil.add_weight(year_text, -0.001)
            | pynutil.add_weight(year_prefix, -0.009)
        )

        self.final_graph = final_graph.optimize()

        self.fst = self.add_tokens(self.final_graph)
