from typing import Literal

from nameparser import HumanName
import gender_guesser.detector as gender


class GenderDetector:
    def __init__(self):
        self.detector = gender.Detector(case_sensitive=False)

    def detect_gender(
        self, name: str
    ) -> Literal["male"] | Literal["female"] | Literal["unknown"]:
        name = HumanName(name)
        first = name.first or name.nickname
        detected = self.detector.get_gender(first)
        if detected in ("male", "female"):
            return detected
        else:
            return "unknown"


def test_gender_detector():
    MALES = [
        "Lucas Mercer",
        "David Miller",
        "Robert K. Wilson",
        "Julian Mercer",
        "christopherhill christopherhill",
        "jonathanwilson jonathanwilson",
        "jordantroyhayes jordantroyhayes",
    ]

    FEMALES = [
        "Sarah Jenkins",
        "marina costa oliveira",
        "elena sofia rodriguez castro",
        "Emily Roberts",
        "sarah m thompson",
        "sarahthompson sarahthompson",
        "amandasmith amandasmith",
        "ms sj williams hv",
        "carol white",
        "fionna holloway",
        "elara vance",
        "elara vance-smith",
    ]

    UNKNOWNS = ["", "visa visa", "cbs cbs", "visadebitserve visadebitserve"]

    DATA = [(MALES, "male"), (FEMALES, "female"), (UNKNOWNS, "unknown")]
    gender_detector = GenderDetector()

    for names, expected in DATA:
        for name in names:
            predicted = gender_detector.detect_gender(name)
            assert expected == predicted, (name, expected, predicted)
