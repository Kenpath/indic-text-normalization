import pynini
from pynini.lib import pynutil

from indic_text_normalization.text_normalization.mag.graph_utils import (
    NEMO_DIGIT,
    NEMO_MAG_DIGIT,
    GraphFst,
    insert_space,
)
from indic_text_normalization.text_normalization.mag.utils import get_abs_path

quantities = pynini.string_file(get_abs_path("data/numbers/thousands.tsv"))

# Convert Arabic digits (0-9) to Magadhi digits (०-९)
arabic_to_magadhi_digit = pynini.string_map([
    ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
    ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९")
]).optimize()
arabic_to_magadhi_number = pynini.closure(arabic_to_magadhi_digit).optimize()


def get_quantity(decimal: 'pynini.FstLike', cardinal_up_to_hundred: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. १ लाख -> integer_part: "एक" quantity: "लाख"
    e.g. १.५ लाख -> integer_part: "एक" fractional_part: "पांच" quantity: "लाख"

    Args:
        decimal: decimal FST
        cardinal_up_to_hundred: cardinal FST
    """
    numbers = cardinal_up_to_hundred

    res = (
        pynutil.insert("integer_part: \"")
        + numbers
        + pynutil.insert("\"")
        + insert_space
        + pynutil.insert("quantity: \"")
        + quantities
        + pynutil.insert("\"")
    )
    res |= decimal + insert_space + pynutil.insert("quantity: \"") + quantities + pynutil.insert("\"")
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g.
        -१२.५००६ अरब -> decimal { negative: "true" integer_part: "बारह"  fractional_part: "पांच शून्य शून्य छ" quantity: "अरब" }
        १ अरब -> decimal { integer_part: "एक" quantity: "अरब" }

    cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        # Support both Magadhi and Arabic digits for fractional part
        # Magadhi digits path: Magadhi digits -> cardinal digit/zero mapping
        magadhi_digit_graph = cardinal.digit | cardinal.zero
        magadhi_fractional_input = pynini.closure(NEMO_MAG_DIGIT, 1)
        magadhi_fractional_graph = pynini.compose(magadhi_fractional_input, magadhi_digit_graph).optimize()
        
        # Arabic digits path: Arabic digits -> convert to Magadhi -> cardinal digit/zero mapping
        arabic_fractional_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_fractional_graph = pynini.compose(
            arabic_fractional_input,
            arabic_to_magadhi_number @ magadhi_digit_graph
        ).optimize()
        
        # Combined fractional digit graph (supports both Magadhi and Arabic digits)
        graph_digit = magadhi_fractional_graph | arabic_fractional_graph
        self.graph = graph_digit + pynini.closure(insert_space + graph_digit).optimize()

        point = pynutil.delete(".")

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + insert_space,
            0,
            1,
        )

        # Support both Magadhi and Arabic digits for integer part
        cardinal_graph = cardinal.final_graph
        
        # Magadhi digits input for integer part
        magadhi_integer_input = pynini.closure(NEMO_MAG_DIGIT, 1)
        magadhi_integer_graph = pynini.compose(magadhi_integer_input, cardinal_graph).optimize()
        
        # Arabic digits input for integer part
        arabic_integer_input = pynini.closure(NEMO_DIGIT, 1)
        arabic_integer_graph = pynini.compose(
            arabic_integer_input,
            arabic_to_magadhi_number @ cardinal_graph
        ).optimize()
        
        # Combined integer graph (supports both Magadhi and Arabic digits)
        integer_graph = magadhi_integer_graph | arabic_integer_graph

        self.graph_fractional = pynutil.insert("fractional_part: \"") + self.graph + pynutil.insert("\"")
        self.graph_integer = pynutil.insert("integer_part: \"") + integer_graph + pynutil.insert("\"")

        final_graph_wo_sign = self.graph_integer + point + insert_space + self.graph_fractional

        # For quantity support, we also need to support both digit types
        cardinal_graph_combined = integer_graph
        self.final_graph_wo_negative = final_graph_wo_sign | get_quantity(final_graph_wo_sign, cardinal_graph_combined)

        final_graph = optional_graph_negative + self.final_graph_wo_negative

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

