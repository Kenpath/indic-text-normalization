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
    GraphFst,
    NEMO_DIGIT,
    NEMO_TA_DIGIT,
    insert_space,
)
from indic_text_normalization.ta.utils import get_abs_path

# Convert Arabic digits (0-9) to Tamil digits (௦-௯)
arabic_to_tamil_digit = pynini.string_map([
    ("0", "௦"), ("1", "௧"), ("2", "௨"), ("3", "௩"), ("4", "௪"),
    ("5", "௫"), ("6", "௬"), ("7", "௭"), ("8", "௮"), ("9", "௯")
]).optimize()
arabic_to_tamil_number = pynini.closure(arabic_to_tamil_digit).optimize()

# Create a graph that deletes commas from digit sequences
# This handles Indian number format where commas are separators (e.g., 1,000,001 or ௧,௦௦௦,௦௦௧)
any_digit = pynini.union(NEMO_DIGIT, NEMO_TA_DIGIT)
delete_commas = (any_digit + pynini.closure(pynini.closure(pynutil.delete(","), 0, 1) + any_digit)).optimize()


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying Tamil cardinals, e.g.
        -௨௩ -> cardinal { negative: "true"  integer: "இருபத்துமூன்று" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        # Load Tamil number mappings efficiently
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
            zero_delete = pynutil.add_weight(pynutil.delete("௦"), -0.1)
            if zeros_counts == 0:
                return digit_graph + suffix
            return digit_graph + (zero_delete**zeros_counts) + suffix

        def create_larger_number_graph(digit_graph, suffix, zeros_counts, sub_graph):
            """Create graph for larger numbers with sub-components"""
            zero_delete = pynutil.add_weight(pynutil.delete("௦"), -0.1)
            if zeros_counts == 0:
                return digit_graph + suffix + insert_space + sub_graph
            return digit_graph + suffix + (zero_delete**zeros_counts) + insert_space + sub_graph

        # Special case: exactly 100 = நூறு
        graph_hundred_exact = hundred_exact
        
        # For 101-109: நூற்று + digit (e.g., 101 = நூற்றொன்று)
        # Pattern: 1 + 0 + digit -> நூற்று + digit
        graph_101_109 = (
            pynutil.delete("௧") + pynutil.delete("௦") + pynutil.insert(" நூற்று") + digit
        )
        
        # For 110-199: நூற்றுப் + tens/teens
        # Pattern: 1 + (10-99) -> நூற்றுப் + tens/teens
        # Note: Special cases 150, 160, 170, 180, 190 are handled separately below
        # They will have priority in the union, so this general pattern won't conflict
        graph_110_199_general = (
            pynutil.delete("௧") + pynutil.insert(" நூற்றுப்") + insert_space + teens_ties
        )
        
        # Special cases for 150, 160, 170, 180, 190: combined forms
        # 150 = நூற்றைம்பது, 160 = நூற்றறுபது, 170 = நூற்றெழுபது, 180 = நூற்றெண்பது, 190 = நூற்றொண்ணூறு
        graph_150 = pynini.cross("௧௫௦", "நூற்றைம்பது")
        graph_160 = pynini.cross("௧௬௦", "நூற்றறுபது")
        graph_170 = pynini.cross("௧௭௦", "நூற்றெழுபது")
        graph_180 = pynini.cross("௧௮௦", "நூற்றெண்பது")
        graph_190 = pynini.cross("௧௯௦", "நூற்றொண்ணூறு")
        graph_150_190 = graph_150 | graph_160 | graph_170 | graph_180 | graph_190
        
        # For 151-159, 161-169, etc.: special combined forms + digit
        # 151 = நூற்றைம்பத்தொன்று, etc.
        graph_151_159 = (
            pynutil.delete("௧") + pynutil.delete("௫") + pynutil.insert("நூற்றைம்பத்து") + digit
        )
        graph_161_169 = (
            pynutil.delete("௧") + pynutil.delete("௬") + pynutil.insert("நூற்றறுபத்து") + digit
        )
        graph_171_179 = (
            pynutil.delete("௧") + pynutil.delete("௭") + pynutil.insert("நூற்றெழுபத்து") + digit
        )
        graph_181_189 = (
            pynutil.delete("௧") + pynutil.delete("௮") + pynutil.insert("நூற்றெண்பத்து") + digit
        )
        graph_191_199 = (
            pynutil.delete("௧") + pynutil.delete("௯") + pynutil.insert("நூற்றொண்ணூற்று") + digit
        )
        graph_151_199_special = graph_151_159 | graph_161_169 | graph_171_179 | graph_181_189 | graph_191_199
        
        # Combine all 100-199 patterns
        # Order matters: special cases first (they're more specific)
        graph_100_199 = (
            graph_hundred_exact
            | graph_101_109
            | graph_150_190  # Special cases first
            | graph_151_199_special  # Special cases first
            | graph_110_199_general  # General pattern last
        )
        
        # For 200-800 exact hundreds: combined forms (இருநூறு, முந்நூறு, etc.)
        # Pattern: digit (2-8) + ௦௦ -> combined_form + நூறு
        tamil_zero = "௦"
        tamil_zero_zero = tamil_zero + tamil_zero
        graph_200_800_exact = (
            hundreds_combined + pynutil.insert("நூறு") + pynutil.delete(tamil_zero_zero)
        )
        
        # Special case: 900 = தொள்ளாயிரம் (not தொள்ளாயிரம் நூறு)
        graph_900_exact = (
            pynini.cross("௯", "தொள்ளாயிரம்") + pynutil.delete(tamil_zero_zero)
        )
        graph_hundreds_exact = graph_200_800_exact | graph_900_exact
        
        # For 201-209: combined_form + நூற்று + digit (e.g., 201 = இருநூற்றொன்று)
        # Pattern: digit (2-8) + ௦ + digit -> combined_form + நூற்று + digit
        graph_201_209_2_8 = (
            hundreds_combined + pynutil.insert("நூற்று") + pynutil.delete(tamil_zero) + digit
        )
        
        # Special case: 901-909: தொள்ளாயிரத்து + digit
        graph_901_909 = (
            pynutil.delete("௯") + pynutil.delete(tamil_zero) + pynutil.insert("தொள்ளாயிரத்து") + digit
        )
        graph_201_209 = graph_201_209_2_8 | graph_901_909
        
        # For 210-299, 310-399, etc. (2-8): combined_form + நூற்றுப் + tens/teens
        # Pattern: digit (2-8) + tens/teens -> combined_form + நூற்றுப் + tens/teens
        graph_210_899 = (
            hundreds_combined + pynutil.insert(" நூற்றுப்") + insert_space + teens_ties
        )
        
        # Special case: 910-999: தொள்ளாயிரத்து + tens/teens
        graph_910_999 = (
            pynutil.delete("௯") + pynutil.insert(" தொள்ளாயிரத்து") + insert_space + teens_ties
        )
        graph_210_999 = graph_210_899 | graph_910_999
        
        # Combine all hundred patterns
        graph_all_hundreds = (
            graph_100_199
            | graph_hundreds_exact
            | graph_201_209
            | graph_210_999
        ).optimize()
        
        self.graph_hundreds = graph_all_hundreds

        # Thousands and Ten thousands graph (1000-99999)
        # Tamil: ஆயிரம் (aayiram)
        suffix_thousands = pynutil.insert(" ஆயிரம்")
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
        # Tamil: இலட்சம் (ilatcham)
        suffix_lakhs = pynutil.insert(" இலட்சம்")
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
        # Tamil: கோடி (kodi)
        suffix_crores = pynutil.insert(" கோடி")
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

        # Handle leading zeros (e.g., 05 -> பூஜ்யம் ஐந்து)
        single_digit = digit | zero
        graph_leading_zero = zero + insert_space + single_digit
        graph_leading_zero = pynutil.add_weight(graph_leading_zero, 0.5)

        # Combine all number patterns efficiently
        # Support both Tamil digits and Arabic digits
        # Tamil digits go directly to final_graph
        tamil_final_graph = (
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
        
        # Arabic digits: convert to Tamil, then apply the same graph
        arabic_digit_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_final_graph = pynini.compose(arabic_digit_input, arabic_to_tamil_number @ tamil_final_graph).optimize()
        
        # Add comma support: delete commas and then interpret as a single number.
        # Tamil with commas
        tamil_with_commas = pynini.compose(delete_commas, tamil_final_graph).optimize()
        tamil_final_with_commas = pynutil.add_weight(tamil_with_commas, -0.1) | tamil_final_graph

        # Arabic with commas: delete commas first, then convert and interpret
        arabic_with_commas = pynini.compose(delete_commas, arabic_to_tamil_number @ tamil_final_graph).optimize()
        arabic_final_with_commas = pynutil.add_weight(arabic_with_commas, -0.1) | arabic_final_graph

        # Combine both Tamil and Arabic digit paths (both with comma support)
        final_graph = tamil_final_with_commas | arabic_final_with_commas

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

