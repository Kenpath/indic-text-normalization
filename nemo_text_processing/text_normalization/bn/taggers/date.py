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

from nemo_text_processing.text_normalization.bn.graph_utils import (
    NEMO_BN_DIGIT,
    NEMO_BN_NON_ZERO,
    NEMO_BN_ZERO,
    NEMO_DIGIT,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.bn.utils import get_abs_path

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

# Convert Arabic digits (0-9) to Bengali digits (০-৯)
arabic_to_bengali_digit = pynini.string_map([
    ("0", "০"), ("1", "১"), ("2", "২"), ("3", "৩"), ("4", "৪"),
    ("5", "৫"), ("6", "৬"), ("7", "৭"), ("8", "৮"), ("9", "৯")
]).optimize()
arabic_to_bengali_number = pynini.closure(arabic_to_bengali_digit).optimize()


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, e.g.
        "০১-০৪-২০২৪" -> date { day: "এক" month: "এপ্রিল" year: "দুই হাজার চব্বিশ" }
        "০৪-০১-২০২৪" -> date { month: "এপ্রিল" day: "এক" year: "দুই হাজার চব্বিশ" }


    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="date", kind="classify")

        # Support both Bengali and Arabic digits for year patterns
        bengali_year_thousands = pynini.compose(
            (NEMO_BN_DIGIT + NEMO_BN_ZERO + NEMO_BN_DIGIT + NEMO_BN_DIGIT), cardinal.graph_thousands
        )
        arabic_year_thousands = pynini.compose(
            (NEMO_DIGIT + pynini.accep("0") + NEMO_DIGIT + NEMO_DIGIT),
            arabic_to_bengali_number @ cardinal.graph_thousands
        )
        graph_year_thousands = bengali_year_thousands | arabic_year_thousands

        bengali_year_hundreds_as_thousands = pynini.compose(
            (NEMO_BN_DIGIT + NEMO_BN_NON_ZERO + NEMO_BN_DIGIT + NEMO_BN_DIGIT), cardinal.graph_hundreds_as_thousand
        )
        arabic_year_hundreds_as_thousands = pynini.compose(
            (NEMO_DIGIT + pynini.union("1", "2", "3", "4", "5", "6", "7", "8", "9") + NEMO_DIGIT + NEMO_DIGIT),
            arabic_to_bengali_number @ cardinal.graph_hundreds_as_thousand
        )
        graph_year_hundreds_as_thousands = bengali_year_hundreds_as_thousands | arabic_year_hundreds_as_thousands

        # Support both Bengali and Arabic digits for cardinal graph
        bengali_cardinal_graph = pynini.union(
            digit, teens_and_ties, cardinal.graph_hundreds, graph_year_thousands, graph_year_hundreds_as_thousands
        )
        arabic_cardinal_graph = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_bengali_number @ bengali_cardinal_graph
        )
        cardinal_graph = bengali_cardinal_graph | arabic_cardinal_graph

        graph_year = pynini.union(graph_year_thousands, graph_year_hundreds_as_thousands)

        delete_dash = pynutil.delete("-")
        delete_slash = pynutil.delete("/")

        # Support both Bengali and Arabic digits for days and months
        bengali_days_graph = pynutil.insert("day: \"") + days + pynutil.insert("\"") + insert_space
        arabic_days_input = pynini.closure(NEMO_DIGIT, 1, 2)
        arabic_days_graph = pynini.compose(
            arabic_days_input,
            arabic_to_bengali_number @ days
        )
        arabic_days_graph = pynutil.insert("day: \"") + arabic_days_graph + pynutil.insert("\"") + insert_space
        days_graph = bengali_days_graph | arabic_days_graph

        bengali_months_graph = pynutil.insert("month: \"") + months + pynutil.insert("\"") + insert_space
        arabic_months_input = pynini.closure(NEMO_DIGIT, 1, 2)
        arabic_months_graph = pynini.compose(
            arabic_months_input,
            arabic_to_bengali_number @ months
        )
        arabic_months_graph = pynutil.insert("month: \"") + arabic_months_graph + pynutil.insert("\"") + insert_space
        months_graph = bengali_months_graph | arabic_months_graph

        years_graph = pynutil.insert("year: \"") + graph_year + pynutil.insert("\"") + insert_space

        graph_dd_mm = days_graph + delete_dash + months_graph

        graph_mm_dd = months_graph + delete_dash + days_graph

        graph_mm_dd += pynutil.insert(" preserve_order: true ")

        # Graph for era
        era_graph = pynutil.insert("era: \"") + year_suffix + pynutil.insert("\"") + insert_space

        range_graph = pynini.cross("-", "থেকে")

        # Graph for year - support both Bengali and Arabic digits
        bengali_century_number = pynini.compose(pynini.closure(NEMO_BN_DIGIT, 1), bengali_cardinal_graph) + pynini.accep("তম")
        arabic_century_number = pynini.compose(
            pynini.closure(NEMO_DIGIT, 1),
            arabic_to_bengali_number @ bengali_cardinal_graph
        ) + pynini.accep("তম")
        century_number = bengali_century_number | arabic_century_number
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

