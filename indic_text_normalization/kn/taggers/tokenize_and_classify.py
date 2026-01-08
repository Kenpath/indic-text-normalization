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

from indic_text_normalization.kn.graph_utils import (
    NEMO_DIGIT,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    NEMO_KN_DIGIT,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from indic_text_normalization.kn.taggers.cardinal import CardinalFst
from indic_text_normalization.kn.taggers.date import DateFst
from indic_text_normalization.kn.taggers.decimal import DecimalFst
from indic_text_normalization.kn.taggers.fraction import FractionFst
from indic_text_normalization.kn.taggers.measure import MeasureFst
from indic_text_normalization.kn.taggers.money import MoneyFst
from indic_text_normalization.kn.taggers.ordinal import OrdinalFst
from indic_text_normalization.kn.taggers.punctuation import PunctuationFst
from indic_text_normalization.kn.taggers.telephone import TelephoneFst
from indic_text_normalization.kn.taggers.time import TimeFst
from indic_text_normalization.kn.taggers.whitelist import WhiteListFst
from indic_text_normalization.kn.taggers.word import WordFst
from indic_text_normalization.kn.taggers.power import PowerFst
from indic_text_normalization.kn.taggers.scientific import ScientificFst
from indic_text_normalization.kn.taggers.serial import SerialFst


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
                f"kn_tn_{deterministic}_deterministic_{input_case}_{whitelist_file}_tokenize.far",
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

            from indic_text_normalization.kn.taggers.math import MathFst
            math = MathFst(cardinal=cardinal, deterministic=deterministic)
            math_graph = math.fst

            power = PowerFst(cardinal=cardinal, deterministic=deterministic)
            power_graph = power.fst

            scientific = ScientificFst(cardinal=cardinal, deterministic=deterministic)
            scientific_graph = scientific.fst

            whitelist_graph = WhiteListFst(
                input_case=input_case, deterministic=deterministic, input_file=whitelist
            ).fst

            punctuation = PunctuationFst(deterministic=deterministic)
            punct_graph = punctuation.fst

            telephone = TelephoneFst()
            telephone_graph = telephone.fst

            serial = SerialFst(cardinal=cardinal, ordinal=ordinal, deterministic=deterministic)
            serial_graph = serial.fst

            # Prioritize telephone numbers and specific patterns over cardinals
            # to avoid incorrect matching (e.g., telephone numbers as cardinals)
            # Lower weight = higher priority in pynini
            classify = (
                pynutil.add_weight(whitelist_graph, 0.5)  # Highest priority
                | pynutil.add_weight(telephone_graph, 0.3)  # High priority for telephone
                | pynutil.add_weight(time_graph, 0.7)  # High priority for time (before cardinals!)
                | pynutil.add_weight(date_graph, 0.8)
                | pynutil.add_weight(decimal_graph, 0.85)
                | pynutil.add_weight(scientific_graph, 0.86)  # 10.1-e5 style
                | pynutil.add_weight(fraction_graph, 0.85)
                | pynutil.add_weight(money_graph, 0.85)
                | pynutil.add_weight(measure_graph, 0.85)
                | pynutil.add_weight(ordinal_graph, 0.9)  # Before cardinals
                | pynutil.add_weight(cardinal_graph, 0.95)  # Before math
                | pynutil.add_weight(power_graph, 0.97)  # Before math: scientific superscripts like 10⁻⁷
                | pynutil.add_weight(math_graph, 1.0)  # Math expressions after cardinals
                | pynutil.add_weight(serial_graph, 1.1)  # Serial numbers
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

            # Replace hyphen used as a joiner between digits and Kannada letters with a SPACE, e.g.
            #   "3.14-ಅಲ್ಲಿ" -> "3.14 ಅಲ್ಲಿ"
            # This prevents "π = 3.1415...-ಅಲ್ಲಿ" from being glued into one token.
            kn_block = pynini.union(*[chr(i) for i in range(0x0C80, 0x0D00)]).optimize()
            left_ctx = pynini.union(NEMO_DIGIT, NEMO_KN_DIGIT).optimize()
            right_ctx = kn_block
            joiner_hyphen_to_space = pynini.cdrewrite(pynini.cross("-", " "), left_ctx, right_ctx, NEMO_SIGMA)

            # Also ensure glued equals patterns like "π=3.1415" tokenize cleanly without enumerating symbols.
            # Only apply when the left side is NOT a digit (so we don't change "10-2=8" tight math behavior).
            non_digit_left = pynini.difference(
                NEMO_NOT_SPACE, pynini.union(NEMO_DIGIT, NEMO_KN_DIGIT)
            ).optimize()
            digit_right = pynini.union(NEMO_DIGIT, NEMO_KN_DIGIT).optimize()
            equals_to_spaced = pynini.cdrewrite(pynini.cross("=", " = "), non_digit_left, digit_right, NEMO_SIGMA)

            # Also separate em-dash glued to a following number, e.g. "—3.14" so decimals can match.
            emdash_to_spaced = pynini.cdrewrite(pynini.cross("—", "— "), "", digit_right, NEMO_SIGMA)

            # And convert em-dash used as a joiner between digits and Kannada letters into a space:
            #   "3.14—ಮತ್ತು" -> "3.14 ಮತ್ತು"
            emdash_joiner_to_space = pynini.cdrewrite(pynini.cross("—", " "), digit_right, kn_block, NEMO_SIGMA)

            self.fst = (emdash_joiner_to_space @ emdash_to_spaced @ equals_to_spaced @ joiner_hyphen_to_space @ graph).optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")

