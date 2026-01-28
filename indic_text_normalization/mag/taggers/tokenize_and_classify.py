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

from indic_text_normalization.mag.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_MAG_DIGIT,
    NEMO_NOT_SPACE,
    NEMO_SPACE,
    NEMO_SIGMA,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from indic_text_normalization.mag.taggers.cardinal import CardinalFst
from indic_text_normalization.mag.taggers.date import DateFst
from indic_text_normalization.mag.taggers.decimal import DecimalFst
from indic_text_normalization.mag.taggers.measure import MeasureFst
from indic_text_normalization.mag.taggers.money import MoneyFst
from indic_text_normalization.mag.taggers.ordinal import OrdinalFst
from indic_text_normalization.mag.taggers.punctuation import PunctuationFst
from indic_text_normalization.mag.taggers.telephone import TelephoneFst
from indic_text_normalization.mag.taggers.time import TimeFst
from indic_text_normalization.mag.taggers.whitelist import WhiteListFst
from indic_text_normalization.mag.taggers.word import WordFst
from indic_text_normalization.mag.taggers.power import PowerFst
from indic_text_normalization.mag.taggers.scientific import ScientificFst


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence including punctuation.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        whitelist: path to a file with whitelist replacements
    """

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
                f"mag_tn_{deterministic}_deterministic_{input_case}_{whitelist_file}_tokenize.far",
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

            timefst = TimeFst(cardinal=cardinal)
            time_graph = timefst.fst

            money = MoneyFst(cardinal=cardinal)
            money_graph = money.fst

            ordinal = OrdinalFst(cardinal=cardinal, deterministic=deterministic)
            ordinal_graph = ordinal.fst

            from indic_text_normalization.mag.taggers.fraction import FractionFst
            fraction = FractionFst(cardinal=cardinal, deterministic=deterministic)
            fraction_graph = fraction.fst

            from indic_text_normalization.mag.taggers.math import MathFst
            math = MathFst(cardinal=cardinal, deterministic=deterministic)
            math_graph = math.fst

            power = PowerFst(cardinal=cardinal, deterministic=deterministic)
            power_graph = power.fst

            scientific = ScientificFst(cardinal=cardinal, deterministic=deterministic)
            scientific_graph = scientific.fst

            date = DateFst(cardinal=cardinal)
            date_graph = date.fst

            telephone = TelephoneFst()
            telephone_graph = telephone.fst

            measure = MeasureFst(cardinal=cardinal, decimal=decimal)
            measure_graph = measure.fst

            whitelist_graph = WhiteListFst(
                input_case=input_case, deterministic=deterministic, input_file=whitelist
            ).fst

            punctuation = PunctuationFst(deterministic=deterministic)
            punct_graph = punctuation.fst

            classify = (
                pynutil.add_weight(whitelist_graph, 1.01)
                | pynutil.add_weight(telephone_graph, 0.9)  # HIGHEST priority for telephone (before cardinal/math)
                | pynutil.add_weight(time_graph, 1.05)  # Higher priority for times
                | pynutil.add_weight(date_graph, 1.05)  # Higher priority for dates
                | pynutil.add_weight(scientific_graph, 1.04)  # Prefer scientific notation
                | pynutil.add_weight(power_graph, 1.045)  # Prefer superscripts like 10⁻⁷
                | pynutil.add_weight(measure_graph, 1.05)  # Higher priority for measures
                | pynutil.add_weight(cardinal_graph, 1.1)
                | pynutil.add_weight(decimal_graph, 1.1)
                | pynutil.add_weight(fraction_graph, 1.1)
                | pynutil.add_weight(money_graph, 1.1)
                | pynutil.add_weight(math_graph, 1.15)  # Math expressions after cardinals
                | pynutil.add_weight(ordinal_graph, 1.1)
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

            # Tokenization rewrites to keep Magahi easily fixable and aligned with Hindi-style behavior:
            # - Joiner hyphens between digits and Devanagari letters -> space
            # - Glue-equals patterns like "π=3.14" -> "π = 3.14" (so "=" can be whitelisted to "बराबर")
            # - Space after em-dash when followed by a number (so decimals match)
            # - Insert space between math symbols (√, ∑, ...) and following digits/letters ("√2" -> "√ 2")
            dev_block = pynini.union(*[chr(i) for i in range(0x0900, 0x0980)]).optimize()
            digit_any = pynini.union(NEMO_DIGIT, NEMO_MAG_DIGIT).optimize()

            joiner_hyphen_to_space = pynini.cdrewrite(pynini.cross("-", " "), digit_any, dev_block, NEMO_SIGMA)

            # Also break joiner hyphens between Latin letters and digits:
            #   "signal-7" -> "signal 7", "7-stage" -> "7 stage"
            # This lets cardinals normalize the digit, while keeping math like "10-2" untouched.
            # EXCLUDE scientific notation: "e-5" or "E-5" should stay together
            e_pattern = pynini.union("e", "E")
            not_scientific_left = pynini.difference(NEMO_ALPHA, e_pattern).optimize()
            latin_to_digit_hyphen = pynini.cdrewrite(pynini.cross("-", " "), not_scientific_left, digit_any, NEMO_SIGMA)
            
            # For digit-to-latin hyphens, EXCLUDE scientific notation patterns like "-e5" or "-E12"
            # (we want "10.1-e5" to stay together for the scientific tagger)
            not_scientific_right = pynini.difference(NEMO_ALPHA, e_pattern).optimize()
            digit_to_latin_hyphen = pynini.cdrewrite(pynini.cross("-", " "), digit_any, not_scientific_right, NEMO_SIGMA)

            non_digit_left = pynini.difference(NEMO_NOT_SPACE, digit_any).optimize()
            equals_to_spaced = pynini.cdrewrite(pynini.cross("=", " = "), non_digit_left, digit_any, NEMO_SIGMA)

            emdash_to_spaced = pynini.cdrewrite(pynini.cross("—", "— "), "", digit_any, NEMO_SIGMA)
            emdash_joiner_to_space = pynini.cdrewrite(pynini.cross("—", " "), digit_any, dev_block, NEMO_SIGMA)

            math_symbols = pynini.union("√", "∑", "∏", "∫", "∬", "∭", "∮", "∂", "∇").optimize()
            following_char = pynini.union(digit_any, NEMO_ALPHA).optimize()
            math_symbol_to_spaced = pynini.cdrewrite(pynutil.insert(" "), math_symbols, following_char, NEMO_SIGMA)

            self.fst = (
                math_symbol_to_spaced
                @ emdash_joiner_to_space
                @ emdash_to_spaced
                @ equals_to_spaced
                @ digit_to_latin_hyphen
                @ latin_to_digit_hyphen
                @ joiner_hyphen_to_space
                @ graph
            ).optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")

