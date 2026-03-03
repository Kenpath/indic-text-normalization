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

from indic_text_normalization.mr.graph_utils import GraphFst, insert_space
from indic_text_normalization.mr.graph_utils import GraphFst, insert_space, NEMO_DIGIT
from indic_text_normalization.mr.utils import get_abs_path

# Convert Arabic digits (0-9) to Marathi digits (०-९)
arabic_to_marathi_digit = pynini.string_map([
    ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
    ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९")
]).optimize()
arabic_to_marathi_number = pynini.closure(arabic_to_marathi_digit, 1).optimize()


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
        -२३ -> cardinal { negative: "true"  integer: "तेवीस" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        digit_file = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        zero_file = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        teens_ties_file = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
        
        # Robust input converter: 
        # 1. Accepts Marathi digits (used as is)
        # 2. Accepts Arabic digits (mapped to Marathi)
        # 3. Deletes commas anywhere
        # Output is always pure Marathi digits string.
        
        # Map Arabic to Marathi OR Identity on Marathi
        utf8_digits = pynini.project(arabic_to_marathi_digit, "output")
        map_digits = arabic_to_marathi_digit | utf8_digits
        
        # Allow commas to be deleted freely around/between digits
        clean_input = pynini.closure(pynutil.delete(","), 0, 1) + map_digits + pynini.closure(pynutil.delete(","), 0, 1)
        # Closure to allow sequences (like 12 for teens)
        clean_input_seq = pynini.closure(map_digits | pynutil.delete(","))

        # Apply to files
        # digit: matches single digit
        digit = clean_input @ digit_file
        zero = clean_input @ zero_file
        
        # teens: matches 2 digits (e.g. 12 or 1,2)
        teens_ties = clean_input_seq @ teens_ties_file
        
        teens_and_ties = pynutil.add_weight(teens_ties, -0.1)

        self.digit = digit
        self.zero = zero
        self.teens_and_ties = teens_and_ties
        
        # Helper to delete zero (0 or ०), handling optional commas
        # Matches "0", "०", ",0", "0,", ",0," etc.
        delete_zero = pynutil.add_weight(pynini.cross("0", "") | pynutil.delete("०"), -0.1)
        # Cleanly allow surrounding commas
        delete_zero_with_comma = pynini.closure(pynutil.delete(","), 0, 1) + delete_zero + pynini.closure(pynutil.delete(","), 0, 1)

        def create_graph_suffix(digit_graph, suffix, zeros_counts):
            if zeros_counts == 0:
                return digit_graph + suffix

            return digit_graph + (delete_zero_with_comma**zeros_counts) + suffix

        def create_larger_number_graph(digit_graph, suffix, zeros_counts, sub_graph):
            insert_space = pynutil.insert(" ")
            if zeros_counts == 0:
                return digit_graph + suffix + insert_space + sub_graph

            return digit_graph + suffix + (delete_zero_with_comma**zeros_counts) + insert_space + sub_graph

        # Hundred graph
        suffix_hundreds = pynutil.insert(" शे")
        hundred_alone = pynini.cross("१००", "शंभर") | pynini.cross("100", "शंभर")
        # Handle 1,00 (comma)? Usually 100 doesn't have comma, but to be safe:
        hundred_alone = pynini.closure(pynutil.delete(","), 0, 1) + hundred_alone + pynini.closure(pynutil.delete(","), 0, 1)
        graph_hundreds = create_graph_suffix(digit, suffix_hundreds, 2)
        graph_hundreds |= create_larger_number_graph(digit, suffix_hundreds, 1, digit)
        graph_hundreds |= create_larger_number_graph(digit, suffix_hundreds, 0, teens_ties)
        graph_hundreds.optimize()
        self.graph_hundreds = graph_hundreds

        # Transducer for eleven hundred -> 1100 or twenty one hundred eleven -> 2111
        graph_hundreds_as_thousand = create_graph_suffix(teens_and_ties, suffix_hundreds, 2)
        graph_hundreds_as_thousand |= create_larger_number_graph(teens_and_ties, suffix_hundreds, 1, digit)
        graph_hundreds_as_thousand |= create_larger_number_graph(teens_and_ties, suffix_hundreds, 0, teens_ties)
        self.graph_hundreds_as_thousand = graph_hundreds_as_thousand

        # Thousands and Ten thousands graph
        suffix_thousands = pynutil.insert(" हजार")
        graph_thousands = create_graph_suffix(digit, suffix_thousands, 3)
        graph_thousands |= create_larger_number_graph(digit, suffix_thousands, 2, digit)
        graph_thousands |= create_larger_number_graph(digit, suffix_thousands, 1, teens_ties)
        graph_thousands |= create_larger_number_graph(digit, suffix_thousands, 0, graph_hundreds)
        graph_thousands.optimize()
        self.graph_thousands = graph_thousands

        graph_ten_thousands = create_graph_suffix(teens_and_ties, suffix_thousands, 3)
        graph_ten_thousands |= create_larger_number_graph(teens_and_ties, suffix_thousands, 2, digit)
        graph_ten_thousands |= create_larger_number_graph(teens_and_ties, suffix_thousands, 1, teens_ties)
        graph_ten_thousands |= create_larger_number_graph(teens_and_ties, suffix_thousands, 0, graph_hundreds)
        graph_ten_thousands.optimize()
        self.graph_ten_thousands = graph_ten_thousands

        # Lakhs graph and ten lakhs graph
        suffix_lakhs = pynutil.insert(" लाख")
        graph_lakhs = create_graph_suffix(digit, suffix_lakhs, 5)
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 4, digit)
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 3, teens_ties)
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 2, graph_hundreds)
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 1, graph_thousands)
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 0, graph_ten_thousands)
        graph_lakhs.optimize()
        self.graph_lakhs = graph_lakhs

        graph_ten_lakhs = create_graph_suffix(teens_and_ties, suffix_lakhs, 5)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 4, digit)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 3, teens_ties)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 2, graph_hundreds)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 1, graph_thousands)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 0, graph_ten_thousands)
        graph_ten_lakhs.optimize()
        self.graph_ten_lakhs = graph_ten_lakhs

        # Crores graph ten crores graph
        suffix_crores = pynutil.insert(" कोटी")
        graph_crores = create_graph_suffix(digit, suffix_crores, 7)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 6, digit)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 5, teens_ties)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 4, graph_hundreds)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 3, graph_thousands)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 2, graph_ten_thousands)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 1, graph_lakhs)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 0, graph_ten_lakhs)
        graph_crores.optimize()

        graph_ten_crores = create_graph_suffix(teens_and_ties, suffix_crores, 7)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 6, digit)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 5, teens_ties)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 4, graph_hundreds)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 3, graph_thousands)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 2, graph_ten_thousands)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 1, graph_lakhs)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 0, graph_ten_lakhs)
        graph_ten_crores.optimize()

        # Arabs graph and ten arabs graph
        suffix_arabs = pynutil.insert(" अब्ज")
        graph_arabs = create_graph_suffix(digit, suffix_arabs, 9)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 8, digit)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 7, teens_ties)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 6, graph_hundreds)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 5, graph_thousands)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 4, graph_ten_thousands)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 3, graph_lakhs)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 2, graph_ten_lakhs)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 1, graph_crores)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 0, graph_ten_crores)
        graph_arabs.optimize()

        graph_ten_arabs = create_graph_suffix(teens_and_ties, suffix_arabs, 9)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 8, digit)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 7, teens_ties)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 6, graph_hundreds)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 5, graph_thousands)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 4, graph_ten_thousands)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 3, graph_lakhs)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 2, graph_ten_lakhs)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 1, graph_crores)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 0, graph_ten_crores)
        graph_ten_arabs.optimize()

        # Kharabs graph and ten kharabs graph
        suffix_kharabs = pynutil.insert(" खर्व")
        graph_kharabs = create_graph_suffix(digit, suffix_kharabs, 11)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 10, digit)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 9, teens_ties)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 8, graph_hundreds)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 7, graph_thousands)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 6, graph_ten_thousands)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 5, graph_lakhs)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 4, graph_ten_lakhs)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 3, graph_crores)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 2, graph_ten_crores)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 1, graph_arabs)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 0, graph_ten_arabs)
        graph_kharabs.optimize()

        graph_ten_kharabs = create_graph_suffix(teens_and_ties, suffix_kharabs, 11)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 10, digit)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 9, teens_ties)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 8, graph_hundreds)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 7, graph_thousands)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 6, graph_ten_thousands)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 5, graph_lakhs)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 4, graph_ten_lakhs)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 3, graph_crores)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 2, graph_ten_crores)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 1, graph_arabs)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 0, graph_ten_arabs)
        graph_ten_kharabs.optimize()

        # Nils graph and ten nils graph
        suffix_nils = pynutil.insert(" निखर्व")
        graph_nils = create_graph_suffix(digit, suffix_nils, 13)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 12, digit)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 11, teens_ties)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 10, graph_hundreds)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 9, graph_thousands)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 8, graph_ten_thousands)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 7, graph_lakhs)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 6, graph_ten_lakhs)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 5, graph_crores)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 4, graph_ten_crores)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 3, graph_arabs)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 2, graph_ten_arabs)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 1, graph_kharabs)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 0, graph_ten_kharabs)
        graph_nils.optimize()

        graph_ten_nils = create_graph_suffix(teens_and_ties, suffix_nils, 13)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 12, digit)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 11, teens_ties)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 10, graph_hundreds)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 9, graph_thousands)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 8, graph_ten_thousands)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 7, graph_lakhs)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 6, graph_ten_lakhs)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 5, graph_crores)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 4, graph_ten_crores)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 3, graph_arabs)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 2, graph_ten_arabs)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 1, graph_kharabs)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 0, graph_ten_kharabs)
        graph_ten_nils.optimize()

        # Padmas graph and ten padmas graph
        suffix_padmas = pynutil.insert(" महापद्म")
        graph_padmas = create_graph_suffix(digit, suffix_padmas, 15)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 14, digit)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 13, teens_ties)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 12, graph_hundreds)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 11, graph_thousands)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 10, graph_ten_thousands)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 9, graph_lakhs)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 8, graph_ten_lakhs)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 7, graph_crores)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 6, graph_ten_crores)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 5, graph_arabs)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 4, graph_ten_arabs)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 3, graph_kharabs)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 2, graph_ten_kharabs)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 1, graph_nils)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 0, graph_ten_nils)
        graph_padmas.optimize()

        graph_ten_padmas = create_graph_suffix(teens_and_ties, suffix_padmas, 15)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 14, digit)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 13, teens_ties)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 12, graph_hundreds)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 11, graph_thousands)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 10, graph_ten_thousands)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 9, graph_lakhs)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 8, graph_ten_lakhs)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 7, graph_crores)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 6, graph_ten_crores)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 5, graph_arabs)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 4, graph_ten_arabs)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 3, graph_kharabs)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 2, graph_ten_kharabs)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 1, graph_nils)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 0, graph_ten_nils)
        graph_ten_padmas.optimize()

        # Shankhs graph and ten shankhs graph
        suffix_shankhs = pynutil.insert(" शंख")
        graph_shankhs = create_graph_suffix(digit, suffix_shankhs, 17)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 16, digit)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 15, teens_ties)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 14, graph_hundreds)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 13, graph_thousands)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 12, graph_ten_thousands)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 11, graph_lakhs)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 10, graph_ten_lakhs)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 9, graph_crores)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 8, graph_ten_crores)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 7, graph_arabs)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 6, graph_ten_arabs)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 5, graph_kharabs)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 4, graph_ten_kharabs)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 3, graph_nils)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 2, graph_ten_nils)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 1, graph_padmas)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 0, graph_ten_padmas)
        graph_shankhs.optimize()

        graph_ten_shankhs = create_graph_suffix(teens_and_ties, suffix_shankhs, 17)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 16, digit)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 15, teens_ties)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 14, graph_hundreds)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 13, graph_thousands)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 12, graph_ten_thousands)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 11, graph_lakhs)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 10, graph_ten_lakhs)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 9, graph_crores)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 8, graph_ten_crores)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 7, graph_arabs)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 6, graph_ten_arabs)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 5, graph_kharabs)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 4, graph_ten_kharabs)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 3, graph_nils)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 2, graph_ten_nils)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 1, graph_padmas)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 0, graph_ten_padmas)
        graph_ten_shankhs.optimize()

        # Only match exactly 2 digits to avoid interfering with telephone numbers, decimals, etc.
        # e.g., "०५" -> "शून्य पाच"
        single_digit = digit | zero
        graph_leading_zero = zero + insert_space + single_digit
        graph_leading_zero = pynutil.add_weight(graph_leading_zero, 0.5)

        final_graph_base = (
            digit
            | zero
            | teens_and_ties
            | hundred_alone
            | graph_hundreds
            | graph_thousands
            | graph_ten_thousands
            | graph_lakhs
            | graph_ten_lakhs
            | graph_crores
            | graph_ten_crores
            | graph_arabs
            | graph_ten_arabs
            | graph_kharabs
            | graph_ten_kharabs
            | graph_nils
            | graph_ten_nils
            | graph_padmas
            | graph_ten_padmas
            | graph_shankhs
            | graph_ten_shankhs
            | graph_leading_zero
        )

        # Comma-aware dual support:
        # - International grouped commas -> million/billion/trillion style.
        # - Indian grouped commas -> existing Indian scale style.
        NEMO_MR_DIGIT = pynini.union("०", "१", "२", "३", "४", "५", "६", "७", "८", "९").optimize()
        any_digit = pynini.union(NEMO_DIGIT, NEMO_MR_DIGIT).optimize()
        comma = pynini.accep(",")
        three_digits = any_digit + any_digit + any_digit
        indian_comma_pattern = (
            pynini.closure(any_digit, 1, 2)
            + pynini.closure(comma + any_digit + any_digit, 1)
            + pynini.closure(comma + three_digits, 0, 1)
        ).optimize()
        delete_commas = (
            any_digit + pynini.closure(pynini.closure(pynutil.delete(","), 0, 1) + any_digit)
        ).optimize()

        mr_1_3 = pynini.closure(NEMO_MR_DIGIT, 1, 3)
        ar_1_3 = pynini.closure(NEMO_DIGIT, 1, 3)
        mr_3 = NEMO_MR_DIGIT + NEMO_MR_DIGIT + NEMO_MR_DIGIT
        ar_3 = NEMO_DIGIT + NEMO_DIGIT + NEMO_DIGIT
        mr_3_nonzero = pynini.difference(mr_3, pynini.accep("०००")).optimize()
        ar_3_nonzero = pynini.difference(ar_3, pynini.accep("000")).optimize()
        up_to_999 = (hundred_alone | graph_hundreds | teens_ties | digit | zero).optimize()
        leading_zero_strip = pynini.closure(pynutil.delete("०"), 0, 2)

        group_1_3 = (
            pynini.compose(mr_1_3, final_graph_base)
            | pynini.compose(ar_1_3, arabic_to_marathi_number @ final_graph_base)
        ).optimize()
        group_3 = (
            pynini.compose(mr_3, leading_zero_strip + up_to_999)
            | pynini.compose(ar_3, arabic_to_marathi_number @ (leading_zero_strip + up_to_999))
        ).optimize()
        group_3_nonzero = (
            pynini.compose(mr_3_nonzero, leading_zero_strip + up_to_999)
            | pynini.compose(ar_3_nonzero, arabic_to_marathi_number @ (leading_zero_strip + up_to_999))
        ).optimize()
        delete_comma = pynutil.delete(",")

        intl_thousand = (group_1_3 + delete_comma + pynutil.insert(" हजार ") + group_3).optimize()
        intl_thousand_zero_tail = (
            group_1_3 + delete_comma + pynutil.insert(" हजार") + pynutil.delete(pynini.union("०००", "000"))
        ).optimize()
        intl_million = (
            group_1_3 + delete_comma + pynutil.insert(" मिलियन ")
            + group_3_nonzero + delete_comma + pynutil.insert(" हजार ") + group_3
        ).optimize()
        intl_million_zero_thousand = (
            group_1_3 + delete_comma + pynutil.insert(" मिलियन ")
            + pynutil.delete(pynini.union("०००", "000")) + delete_comma + group_3
        ).optimize()
        intl_billion = (
            group_1_3 + delete_comma + pynutil.insert(" बिलियन ")
            + group_3_nonzero + delete_comma + pynutil.insert(" मिलियन ")
            + group_3_nonzero + delete_comma + pynutil.insert(" हजार ") + group_3
        ).optimize()
        intl_trillion = (
            group_1_3 + delete_comma + pynutil.insert(" ट्रिलियन ")
            + group_3_nonzero + delete_comma + pynutil.insert(" बिलियन ")
            + group_3_nonzero + delete_comma + pynutil.insert(" मिलियन ")
            + group_3_nonzero + delete_comma + pynutil.insert(" हजार ") + group_3
        ).optimize()
        strict_intl_with_commas = (
            pynutil.add_weight(intl_million_zero_thousand, -0.1)
            | pynutil.add_weight(intl_thousand_zero_tail, -0.1)
            | intl_trillion
            | intl_billion
            | intl_million
            | intl_thousand
        ).optimize()

        mr_with_commas = (pynini.compose(indian_comma_pattern, delete_commas) @ final_graph_base).optimize()
        arabic_with_commas = (
            pynini.compose(indian_comma_pattern, delete_commas) @ arabic_to_marathi_number @ final_graph_base
        ).optimize()

        final_graph = (
            pynutil.add_weight(strict_intl_with_commas, -0.1)
            | pynutil.add_weight(mr_with_commas, -0.1)
            | pynutil.add_weight(arabic_with_commas, -0.1)
            | final_graph_base
        )

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        self.final_graph = final_graph.optimize()
        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.final_graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph
