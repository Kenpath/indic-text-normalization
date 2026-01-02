import sys
from unicodedata import category

import pynini
from pynini.examples import plurals
from pynini.lib import pynutil

from indic_text_normalization.te.graph_utils import NEMO_NOT_SPACE, NEMO_SIGMA, GraphFst
from indic_text_normalization.te.utils import get_abs_path


class PunctuationFst(GraphFst):
    """
    Finite state transducer for classifying punctuation
        e.g. a, -> tokens { name: "a" } tokens { name: "," }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transductions are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="punctuation", kind="classify", deterministic=deterministic)
        s = "!#%&\'()*+,-./:;<=>?@^_`{|}~\""

        punct_symbols_to_exclude = ["[", "]"]
        punct_unicode = [
            chr(i)
            for i in range(sys.maxunicode)
            if category(chr(i)).startswith("P") and chr(i) not in punct_symbols_to_exclude
        ]

        self.punct_marks = [p for p in punct_unicode + list(s)]

        punct = pynini.union(*self.punct_marks)
        punct = pynini.closure(punct, 1)

        # Verbalize "=" everywhere (not only inside MathFst) using the shared math operator mapping.
        math_operations = pynini.string_file(get_abs_path("data/math_operations.tsv"))
        equals_spoken = pynini.union("=") @ math_operations
        punct = plurals._priority_union(equals_spoken, punct, NEMO_SIGMA)

        emphasis = (
            pynini.accep("<")
            + pynini.union(
                (pynini.closure(NEMO_NOT_SPACE - pynini.union("<", ">"), 1) + pynini.closure(pynini.accep("/"), 0, 1)),
                (pynini.accep("/") + pynini.closure(NEMO_NOT_SPACE - pynini.union("<", ">"), 1)),
            )
            + pynini.accep(">")
        )
        punct = plurals._priority_union(emphasis, punct, NEMO_SIGMA)

        self.graph = punct
        self.fst = (pynutil.insert("name: \"") + self.graph + pynutil.insert("\"")).optimize()

