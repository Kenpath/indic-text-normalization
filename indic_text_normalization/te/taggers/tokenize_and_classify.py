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

import logging
import os

import pynini
from pynini.lib import pynutil

from indic_text_normalization.te.graph_utils import (
    NEMO_DIGIT,
    NEMO_TE_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    NEMO_ALPHA,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from indic_text_normalization.te.taggers.cardinal import CardinalFst
from indic_text_normalization.te.taggers.date import DateFst
from indic_text_normalization.te.taggers.decimal import DecimalFst
from indic_text_normalization.te.taggers.fraction import FractionFst
from indic_text_normalization.te.taggers.measure import MeasureFst
from indic_text_normalization.te.taggers.money import MoneyFst
from indic_text_normalization.te.taggers.ordinal import OrdinalFst
from indic_text_normalization.te.taggers.power import PowerFst
from indic_text_normalization.te.taggers.punctuation import PunctuationFst
from indic_text_normalization.te.taggers.scientific import ScientificFst
from indic_text_normalization.te.taggers.telephone import TelephoneFst
from indic_text_normalization.te.taggers.time import TimeFst
from indic_text_normalization.te.taggers.whitelist import WhiteListFst
from indic_text_normalization.te.taggers.word import WordFst


from indic_text_normalization.te.taggers.serial import SerialFst

class ClassifyFst(GraphFst):
    def __init__(
        self,
        input_case: str,
        deterministic: bool = True,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            whitelist_file = os.path.basename(whitelist) if whitelist else ""
            far_file = os.path.join(
                cache_dir,
                f"te_tn_{deterministic}_deterministic_{input_case}_{whitelist_file}_tokenize.far",
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logging.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logging.info(f"Creating ClassifyFst grammars.")

            cardinal = CardinalFst(deterministic=deterministic)
            cardinal_graph = cardinal.fst

            decimal = DecimalFst(cardinal=cardinal, deterministic=deterministic)
            decimal_graph = decimal.fst

            fraction = FractionFst(cardinal=cardinal, deterministic=deterministic)
            fraction_graph = fraction.fst

            date = DateFst(cardinal=cardinal)
            date_graph = date.fst

            timefst = TimeFst(cardinal=cardinal)
            time_graph = timefst.fst

            measure = MeasureFst(cardinal=cardinal, decimal=decimal)
            measure_graph = measure.fst

            money = MoneyFst(cardinal=cardinal)
            money_graph = money.fst

            ordinal = OrdinalFst(cardinal=cardinal, deterministic=deterministic)
            ordinal_graph = ordinal.fst

            from indic_text_normalization.te.taggers.math import MathFst
            math = MathFst(cardinal=cardinal, deterministic=deterministic)
            math_graph = math.fst

            power = PowerFst(cardinal=cardinal, deterministic=deterministic)
            power_graph = power.fst

            scientific = ScientificFst(cardinal=cardinal, deterministic=deterministic)
            scientific_graph = scientific.fst

            serial = SerialFst(cardinal=cardinal, ordinal=ordinal, deterministic=deterministic)
            serial_graph = serial.fst

            whitelist_graph = WhiteListFst(
                input_case=input_case, deterministic=deterministic, input_file=whitelist
            ).fst

            punctuation = PunctuationFst(deterministic=deterministic)
            punct_graph = punctuation.fst

            telephone = TelephoneFst()
            telephone_graph = telephone.fst

            # Prioritize telephone numbers and specific patterns over cardinals
            # to avoid incorrect matching (e.g., telephone numbers as cardinals)
            # Lower weight = higher priority in pynini
            classify = (
                pynutil.add_weight(whitelist_graph, 0.5)  # Highest priority
                | pynutil.add_weight(telephone_graph, 0.3)  # High priority for telephone
                | pynutil.add_weight(time_graph, 0.7)  # High priority for time (before cardinals!)
                | pynutil.add_weight(date_graph, 0.8)
                | pynutil.add_weight(decimal_graph, 0.85)
                | pynutil.add_weight(fraction_graph, 0.85)
                | pynutil.add_weight(money_graph, 0.85)
                | pynutil.add_weight(measure_graph, 0.85)
                | pynutil.add_weight(ordinal_graph, 0.9)  # Before cardinals
                | pynutil.add_weight(cardinal_graph, 0.95)  # Before math
                | pynutil.add_weight(math_graph, 1.0)  # Math expressions after cardinals
                | pynutil.add_weight(scientific_graph, 1.08)  # Higher priority for scientific notation
                | pynutil.add_weight(power_graph, 1.09)  # Higher priority for superscripts
                | pynutil.add_weight(serial_graph, 1.12)  # Serial numbers
            )

            word_graph = WordFst(punctuation=punctuation, deterministic=deterministic).fst

            punct = pynutil.insert("tokens { ") + pynutil.add_weight(punct_graph, weight=2.1) + pynutil.insert(" }")
            punct = pynini.closure(
                pynini.union(
                    pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space),
                    (pynutil.insert(NEMO_SPACE) + punct),
                ),
                1,
            )

            classify = pynini.union(classify, pynutil.add_weight(word_graph, 100))
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(NEMO_SPACE))
                + token
                + pynini.closure(pynutil.insert(NEMO_SPACE) + punct)
            )

            graph = token_plus_punct + pynini.closure(
                pynini.union(
                    pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space),
                    (pynutil.insert(NEMO_SPACE) + punct + pynutil.insert(NEMO_SPACE)),
                )
                + token_plus_punct
            )

            graph = delete_space + graph + delete_space
            graph = pynini.union(graph, punct)

            # Replace hyphen used as a joiner between digits and Telugu letters with a SPACE, e.g.
            #   "3.14-అక్కడ" -> "3.14 అక్కడ"
            # This prevents "π = 3.1415...-అక్కడ" from being glued into one token.
            te_block = pynini.union(*[chr(i) for i in range(0x0C00, 0x0C80)]).optimize()
            left_ctx = pynini.union(NEMO_DIGIT, NEMO_TE_DIGIT).optimize()
            right_ctx = te_block
            joiner_hyphen_to_space = pynini.cdrewrite(pynini.cross("-", " "), left_ctx, right_ctx, NEMO_SIGMA)

            # Insert space between mathematical symbols (√, ∑, ∫, etc.) and following digits/letters
            # Example: "√2" -> "√ 2", "∑x" -> "∑ x"
            math_symbols = pynini.union("√", "∑", "∏", "∫", "∬", "∭", "∮", "∂", "∇").optimize()
            following_char = pynini.union(NEMO_DIGIT, NEMO_TE_DIGIT, NEMO_ALPHA).optimize()
            math_symbol_to_spaced = pynini.cdrewrite(pynutil.insert(" "), math_symbols, following_char, NEMO_SIGMA)

            self.fst = (math_symbol_to_spaced @ joiner_hyphen_to_space @ graph).optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")

