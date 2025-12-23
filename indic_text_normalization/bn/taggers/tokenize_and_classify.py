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

from indic_text_normalization.bn.graph_utils import (
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from indic_text_normalization.bn.taggers.cardinal import CardinalFst
from indic_text_normalization.bn.taggers.date import DateFst
from indic_text_normalization.bn.taggers.decimal import DecimalFst
from indic_text_normalization.bn.taggers.fraction import FractionFst
from indic_text_normalization.bn.taggers.money import MoneyFst
from indic_text_normalization.bn.taggers.ordinal import OrdinalFst
from indic_text_normalization.bn.taggers.punctuation import PunctuationFst
from indic_text_normalization.bn.taggers.telephone import TelephoneFst
from indic_text_normalization.bn.taggers.time import TimeFst
from indic_text_normalization.bn.taggers.whitelist import WhiteListFst
from indic_text_normalization.bn.taggers.word import WordFst


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
                f"bn_tn_{deterministic}_deterministic_{input_case}_{whitelist_file}_tokenize.far",
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

            money = MoneyFst(cardinal=cardinal)
            money_graph = money.fst

            ordinal = OrdinalFst(cardinal=cardinal, deterministic=deterministic)
            ordinal_graph = ordinal.fst

            from indic_text_normalization.bn.taggers.math import MathFst
            math = MathFst(cardinal=cardinal, deterministic=deterministic)
            math_graph = math.fst

            whitelist_graph = WhiteListFst(
                input_case=input_case, deterministic=deterministic, input_file=whitelist
            ).fst

            punctuation = PunctuationFst(deterministic=deterministic)
            punct_graph = punctuation.fst

            telephone = TelephoneFst()
            telephone_graph = telephone.fst

            classify = (
                pynutil.add_weight(whitelist_graph, 1.01)
                | pynutil.add_weight(telephone_graph, 1.02)  # Telephone highest priority (numbers with dashes)
                | pynutil.add_weight(date_graph, 1.04)  # Date before time (dates with separators)
                | pynutil.add_weight(time_graph, 1.05)  # Higher priority for times
                | pynutil.add_weight(fraction_graph, 1.06)  # Fraction before cardinal (3/4 should be fraction, not cardinal)
                | pynutil.add_weight(decimal_graph, 1.08)  # Decimal before cardinal
                | pynutil.add_weight(cardinal_graph, 1.1)
                | pynutil.add_weight(money_graph, 1.1)
                | pynutil.add_weight(ordinal_graph, 1.1)
                | pynutil.add_weight(math_graph, 1.2)  # Math expressions lowest priority (avoid matching dashes as minus)
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

            self.fst = graph.optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")

