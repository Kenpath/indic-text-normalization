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

from indic_text_normalization.bho.graph_utils import (
    GraphFst,
    NEMO_DIGIT,
    NEMO_BHO_DIGIT,
    insert_space,
)
from indic_text_normalization.bho.utils import get_abs_path

# Convert Arabic digits (0-9) to Bhojpuri digits (०-९)
arabic_to_bhojpuri_digit = pynini.string_map([
    ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
    ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९")
]).optimize()
arabic_to_bhojpuri_number = pynini.closure(arabic_to_bhojpuri_digit).optimize()


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying Bhojpuri cardinals, e.g.
        -२३ -> cardinal { negative: "true"  integer: "तेईस" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        # Load Bhojpuri number mappings efficiently
        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).optimize()
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).optimize()
        teens_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv")).optimize()
        teens_and_ties = pynutil.add_weight(teens_ties, -0.1)
        
        # Load special hundred forms (200-900 combined forms)
        hundreds_combined = pynini.string_file(get_abs_path("data/numbers/hundreds_combined.tsv")).optimize()
        hundred_exact = pynini.string_file(get_abs_path("data/numbers/hundred.tsv")).optimize()

        self.digit = digit
        self.zero = zero
        self.teens_and_ties = teens_and_ties

        # Helper function to create graphs with zero padding efficiently
        def create_graph_suffix(digit_graph, suffix, zeros_counts):
            """Create graph with suffix and zero padding"""
            zero_delete = pynutil.add_weight(pynutil.delete("०"), -0.1)
            if zeros_counts == 0:
                return digit_graph + suffix
            return digit_graph + (zero_delete**zeros_counts) + suffix

        def create_larger_number_graph(digit_graph, suffix, zeros_counts, sub_graph):
            """Create graph for larger numbers with sub-components"""
            zero_delete = pynutil.add_weight(pynutil.delete("०"), -0.1)
            if zeros_counts == 0:
                return digit_graph + suffix + insert_space + sub_graph
            return digit_graph + suffix + (zero_delete**zeros_counts) + insert_space + sub_graph

        # Special case: exactly 100 = सौ
        graph_hundred_exact = hundred_exact
        
        # For 101-109: सौ + digit (e.g., 101 = एक सौ एक)
        # Pattern: 1 + 0 + digit -> सौ + digit
        graph_101_109 = (
            pynutil.delete("१") + pynutil.delete("०") + pynutil.insert(" सौ") + insert_space + digit
        )
        
        # For 110-199: सौ + tens/teens
        # Pattern: 1 + (10-99) -> सौ + tens/teens
        graph_110_199_general = (
            pynutil.delete("१") + pynutil.insert(" सौ") + insert_space + teens_ties
        )
        
        # Special cases for 150, 160, 170, 180, 190
        graph_150 = pynini.cross("१५०", "एक सौ पचास")
        graph_160 = pynini.cross("१६०", "एक सौ साठ")
        graph_170 = pynini.cross("१७०", "एक सौ सत्तर")
        graph_180 = pynini.cross("१८०", "एक सौ अस्सी")
        graph_190 = pynini.cross("१९०", "एक सौ नब्बे")
        graph_150_190 = graph_150 | graph_160 | graph_170 | graph_180 | graph_190
        
        # For 151-159, 161-169, etc.: special combined forms + digit
        graph_151_159 = (
            pynutil.delete("१") + pynutil.delete("५") + pynutil.insert("एक सौ पचास") + insert_space + digit
        )
        graph_161_169 = (
            pynutil.delete("१") + pynutil.delete("६") + pynutil.insert("एक सौ साठ") + insert_space + digit
        )
        graph_171_179 = (
            pynutil.delete("१") + pynutil.delete("७") + pynutil.insert("एक सौ सत्तर") + insert_space + digit
        )
        graph_181_189 = (
            pynutil.delete("१") + pynutil.delete("८") + pynutil.insert("एक सौ अस्सी") + insert_space + digit
        )
        graph_191_199 = (
            pynutil.delete("१") + pynutil.delete("९") + pynutil.insert("एक सौ नब्बे") + insert_space + digit
        )
        graph_151_199_special = graph_151_159 | graph_161_169 | graph_171_179 | graph_181_189 | graph_191_199
        
        # Combine all 100-199 patterns
        graph_100_199 = (
            graph_hundred_exact
            | graph_101_109
            | graph_150_190
            | graph_151_199_special
            | graph_110_199_general
        )
        
        # For 200-900 exact hundreds: combined forms
        bhojpuri_zero = "०"
        bhojpuri_zero_zero = bhojpuri_zero + bhojpuri_zero
        graph_200_900_exact = (
            hundreds_combined + pynutil.insert(" सौ") + pynutil.delete(bhojpuri_zero_zero)
        )
        
        # For 201-209: combined_form + digit
        graph_201_209 = (
            hundreds_combined + pynutil.insert(" सौ") + insert_space + pynutil.delete(bhojpuri_zero) + digit
        )
        
        # For 210-999: combined_form + tens/teens
        graph_210_999 = (
            hundreds_combined + pynutil.insert(" सौ") + insert_space + teens_ties
        )
        
        # Combine all hundred patterns
        graph_all_hundreds = (
            graph_100_199
            | graph_200_900_exact
            | graph_201_209
            | graph_210_999
        ).optimize()
        
        self.graph_hundreds = graph_all_hundreds

        # Thousands and Ten thousands graph (1000-99999)
        # Bhojpuri: हज़ार (hazaar)
        suffix_thousands = pynutil.insert(" हज़ार")
        graph_thousands = create_graph_suffix(digit, suffix_thousands, 3)
        graph_thousands |= create_larger_number_graph(digit, suffix_thousands, 2, digit)
        graph_thousands |= create_larger_number_graph(digit, suffix_thousands, 1, teens_ties)
        graph_thousands |= create_larger_number_graph(digit, suffix_thousands, 0, graph_all_hundreds)
        graph_thousands.optimize()
        self.graph_thousands = graph_thousands

        graph_ten_thousands = create_graph_suffix(teens_and_ties, suffix_thousands, 3)
        graph_ten_thousands |= create_larger_number_graph(teens_and_ties, suffix_thousands, 2, digit)
        graph_ten_thousands |= create_larger_number_graph(teens_and_ties, suffix_thousands, 1, teens_ties)
        graph_ten_thousands |= create_larger_number_graph(teens_and_ties, suffix_thousands, 0, graph_all_hundreds)
        graph_ten_thousands.optimize()
        self.graph_ten_thousands = graph_ten_thousands

        # Lakhs graph and ten lakhs graph (100000-9999999)
        # Bhojpuri: लाख (laakh)
        suffix_lakhs = pynutil.insert(" लाख")
        graph_lakhs = create_graph_suffix(digit, suffix_lakhs, 5)
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 4, digit)
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 3, teens_ties)
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 2, graph_all_hundreds)
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 1, graph_thousands)
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 0, graph_ten_thousands)
        graph_lakhs.optimize()
        self.graph_lakhs = graph_lakhs

        graph_ten_lakhs = create_graph_suffix(teens_and_ties, suffix_lakhs, 5)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 4, digit)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 3, teens_ties)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 2, graph_all_hundreds)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 1, graph_thousands)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 0, graph_ten_thousands)
        graph_ten_lakhs.optimize()
        self.graph_ten_lakhs = graph_ten_lakhs

        # Crores graph and ten crores graph (10000000+)
        # Bhojpuri: करोड़ (karod)
        suffix_crores = pynutil.insert(" करोड़")
        graph_crores = create_graph_suffix(digit, suffix_crores, 7)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 6, digit)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 5, teens_ties)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 4, graph_all_hundreds)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 3, graph_thousands)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 2, graph_ten_thousands)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 1, graph_lakhs)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 0, graph_ten_lakhs)
        graph_crores.optimize()
        self.graph_crores = graph_crores

        graph_ten_crores = create_graph_suffix(teens_and_ties, suffix_crores, 7)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 6, digit)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 5, teens_ties)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 4, graph_all_hundreds)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 3, graph_thousands)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 2, graph_ten_thousands)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 1, graph_lakhs)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 0, graph_ten_lakhs)
        graph_ten_crores.optimize()
        self.graph_ten_crores = graph_ten_crores

        # Handle leading zeros (e.g., 05 -> शून्य पाँच)
        single_digit = digit | zero
        graph_leading_zero = zero + insert_space + single_digit
        graph_leading_zero = pynutil.add_weight(graph_leading_zero, 0.5)

        # Combine all number patterns efficiently
        # Support both Bhojpuri digits and Arabic digits
        # Bhojpuri digits go directly to final_graph
        bhojpuri_final_graph = (
            digit
            | zero
            | teens_and_ties
            | graph_all_hundreds
            | graph_thousands
            | graph_ten_thousands
            | graph_lakhs
            | graph_ten_lakhs
            | graph_crores
            | graph_ten_crores
            | graph_leading_zero
        ).optimize()
        
        # Arabic digits: convert to Bhojpuri, then apply the same graph
        arabic_digit_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_final_graph = pynini.compose(arabic_digit_input, arabic_to_bhojpuri_number @ bhojpuri_final_graph).optimize()
        
        # Combine both Bhojpuri and Arabic digit paths
        final_graph = bhojpuri_final_graph | arabic_final_graph

        # Handle negative numbers
        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1
        )

        self.final_graph = final_graph
        final_graph = (
            optional_minus_graph
            + pynutil.insert("integer: \"")
            + self.final_graph
            + pynutil.insert("\"")
        )
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

