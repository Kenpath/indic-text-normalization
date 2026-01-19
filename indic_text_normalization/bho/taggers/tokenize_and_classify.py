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
import time

import pynini
from pynini.lib import pynutil

from indic_text_normalization.bho.graph_utils import (
    NEMO_DIGIT,
    NEMO_BHO_DIGIT,
    NEMO_ALPHA,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from indic_text_normalization.bho.taggers.cardinal import CardinalFst
from indic_text_normalization.bho.taggers.date import DateFst
from indic_text_normalization.bho.taggers.decimal import DecimalFst
from indic_text_normalization.bho.taggers.fraction import FractionFst
from indic_text_normalization.bho.taggers.measure import MeasureFst
from indic_text_normalization.bho.taggers.money import MoneyFst
from indic_text_normalization.bho.taggers.ordinal import OrdinalFst
from indic_text_normalization.bho.taggers.punctuation import PunctuationFst
from indic_text_normalization.bho.taggers.telephone import TelephoneFst
from indic_text_normalization.bho.taggers.time import TimeFst
from indic_text_normalization.bho.taggers.whitelist import WhiteListFst
from indic_text_normalization.bho.taggers.word import WordFst
from indic_text_normalization.bho.taggers.power import PowerFst
from indic_text_normalization.bho.taggers.scientific import ScientificFst


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
                f"bho_tn_{deterministic}_deterministic_{input_case}_{whitelist_file}_tokenize.far",
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logging.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logging.info(f"Creating ClassifyFst grammars.")

            start_time = time.time()
            cardinal = CardinalFst(deterministic=deterministic)
            cardinal_graph = cardinal.fst
            logging.debug(f"cardinal: {time.time() - start_time:.2f}s -- {cardinal_graph.num_states()} nodes")

            start_time = time.time()
            decimal = DecimalFst(cardinal=cardinal, deterministic=deterministic)
            decimal_graph = decimal.fst
            logging.debug(f"decimal: {time.time() - start_time:.2f}s -- {decimal_graph.num_states()} nodes")

            start_time = time.time()
            fraction = FractionFst(cardinal=cardinal, deterministic=deterministic)
            fraction_graph = fraction.fst
            logging.debug(f"fraction: {time.time() - start_time:.2f}s -- {fraction_graph.num_states()} nodes")

            start_time = time.time()
            timefst = TimeFst(cardinal=cardinal)
            time_graph = timefst.fst
            logging.debug(f"time: {time.time() - start_time:.2f}s -- {time_graph.num_states()} nodes")

            start_time = time.time()
            datefst = DateFst(cardinal=cardinal)
            date_graph = datefst.fst
            logging.debug(f"date: {time.time() - start_time:.2f}s -- {date_graph.num_states()} nodes")

            start_time = time.time()
            money = MoneyFst(cardinal=cardinal)
            money_graph = money.fst
            logging.debug(f"money: {time.time() - start_time:.2f}s -- {money_graph.num_states()} nodes")

            start_time = time.time()
            measure = MeasureFst(cardinal=cardinal, decimal=decimal, deterministic=deterministic)
            measure_graph = measure.fst
            logging.debug(f"measure: {time.time() - start_time:.2f}s -- {measure_graph.num_states()} nodes")

            start_time = time.time()
            telephone = TelephoneFst(deterministic=deterministic)
            telephone_graph = telephone.fst
            logging.debug(f"telephone: {time.time() - start_time:.2f}s -- {telephone_graph.num_states()} nodes")

            start_time = time.time()
            ordinal = OrdinalFst(cardinal=cardinal, deterministic=deterministic)
            ordinal_graph = ordinal.fst
            logging.debug(f"ordinal: {time.time() - start_time:.2f}s -- {ordinal_graph.num_states()} nodes")

            start_time = time.time()
            from indic_text_normalization.bho.taggers.math import MathFst
            math = MathFst(cardinal=cardinal, deterministic=deterministic)
            math_graph = math.fst
            logging.debug(f"math: {time.time() - start_time:.2f}s -- {math_graph.num_states()} nodes")

            start_time = time.time()
            power = PowerFst(cardinal=cardinal, deterministic=deterministic)
            power_graph = power.fst
            logging.debug(f"power: {time.time() - start_time:.2f}s -- {power_graph.num_states()} nodes")

            start_time = time.time()
            scientific = ScientificFst(cardinal=cardinal, deterministic=deterministic)
            scientific_graph = scientific.fst
            logging.debug(f"scientific: {time.time() - start_time:.2f}s -- {scientific_graph.num_states()} nodes")

            start_time = time.time()
            whitelist_graph = WhiteListFst(
                input_case=input_case, deterministic=deterministic, input_file=whitelist
            ).fst
            logging.debug(f"whitelist: {time.time() - start_time:.2f}s -- {whitelist_graph.num_states()} nodes")

            start_time = time.time()
            punctuation = PunctuationFst(deterministic=deterministic)
            punct_graph = punctuation.fst
            logging.debug(f"punct: {time.time() - start_time:.2f}s -- {punct_graph.num_states()} nodes")

            # Weight ordering - lower weight = higher priority
            classify = (
                pynutil.add_weight(whitelist_graph, 1.01)
                | pynutil.add_weight(telephone_graph, 1.02)  # Telephone highest priority (numbers with dashes)
                | pynutil.add_weight(measure_graph, 1.03)  # Measure before date/time (12 kg should be measure)
                | pynutil.add_weight(date_graph, 1.04)  # Date before time (dates with separators)
                | pynutil.add_weight(time_graph, 1.05)  # Higher priority for times
                | pynutil.add_weight(fraction_graph, 1.06)  # Fraction before cardinal (3/4 should be fraction, not cardinal)
                | pynutil.add_weight(decimal_graph, 1.08)
                | pynutil.add_weight(scientific_graph, 1.08)  # Higher priority for scientific notation
                | pynutil.add_weight(power_graph, 1.09)  # Higher priority for superscripts
                | pynutil.add_weight(cardinal_graph, 1.1)
                | pynutil.add_weight(money_graph, 1.1)
                | pynutil.add_weight(ordinal_graph, 1.1)
                | pynutil.add_weight(math_graph, 1.2)  # Math expressions lowest priority (avoid matching dashes as minus)
            )

            start_time = time.time()
            word_graph = WordFst(punctuation=punctuation, deterministic=deterministic).fst
            logging.debug(f"word: {time.time() - start_time:.2f}s -- {word_graph.num_states()} nodes")

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

            # Rewrite joiner hyphens between digits and Bhojpuri/Devanagari letters to spaces.
            # Example: "3.14-वहाँ" -> "3.14 वहाँ"
            # This prevents "π = 3.1415...-वहाँ" from being glued into one token.
            devanagari_block = pynini.union(*[chr(i) for i in range(0x0900, 0x0980)]).optimize()
            left_ctx = pynini.union(NEMO_DIGIT, NEMO_BHO_DIGIT).optimize()
            right_ctx = devanagari_block
            joiner_hyphen_to_space = pynini.cdrewrite(pynini.cross("-", " "), left_ctx, right_ctx, NEMO_SIGMA)

            # Also handle underscore as a joiner
            joiner_underscore_to_space = pynini.cdrewrite(pynini.cross("_", " "), left_ctx, right_ctx, NEMO_SIGMA)

            start_time = time.time()
            # Also ensure glued equals patterns like "π=3.1415" tokenize cleanly without enumerating symbols.
            # Only apply when the left side is NOT a digit (so we don't change "10-2=8" tight math behavior).
            non_digit_left = pynini.difference(
                NEMO_NOT_SPACE, pynini.union(NEMO_DIGIT, NEMO_BHO_DIGIT)
            ).optimize()
            digit_right = pynini.union(NEMO_DIGIT, NEMO_BHO_DIGIT).optimize()
            equals_to_spaced = pynini.cdrewrite(pynini.cross("=", " = "), non_digit_left, digit_right, NEMO_SIGMA)

            # Also separate em-dash glued to a following number, e.g. "—3.14" so decimals can match.
            emdash_to_spaced = pynini.cdrewrite(pynini.cross("—", "— "), "", digit_right, NEMO_SIGMA)

            # And convert em-dash used as a joiner between digits and Devanagari letters into a space:
            #   "3.14—और" -> "3.14 और"
            emdash_joiner_to_space = pynini.cdrewrite(pynini.cross("—", " "), digit_right, devanagari_block, NEMO_SIGMA)

            # Insert space when digits are directly followed by Devanagari letters (no separator)
            # Example: "3.14159265358979वहाँ" -> "3.14159265358979 वहाँ"
            # This handles cases where numbers are glued directly to text
            all_digits = pynini.union(NEMO_DIGIT, NEMO_BHO_DIGIT).optimize()
            digit_indic_insert_space = pynini.cdrewrite(pynutil.insert(" "), all_digits, devanagari_block, NEMO_SIGMA)

            # Insert space between mathematical symbols (√, ∑, ∫, etc.) and following digits/letters
            # Example: "√2" -> "√ 2", "∑x" -> "∑ x"
            math_symbols = pynini.union("√", "∑", "∏", "∫", "∬", "∭", "∮", "∂", "∇").optimize()
            following_char = pynini.union(NEMO_DIGIT, NEMO_BHO_DIGIT, NEMO_ALPHA).optimize()
            math_symbol_to_spaced = pynini.cdrewrite(pynutil.insert(" "), math_symbols, following_char, NEMO_SIGMA)

            # Apply preprocessing: direct digit-letter attachment first, then specific separators
            self.fst = (math_symbol_to_spaced @ digit_indic_insert_space @ emdash_joiner_to_space @ emdash_to_spaced @ equals_to_spaced @ joiner_underscore_to_space @ joiner_hyphen_to_space @ graph).optimize()
            logging.debug(f"final graph optimization: {time.time() - start_time:.2f}s -- {self.fst.num_states()} nodes")

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")
