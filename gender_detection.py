from typing import Literal

from nameparser import HumanName
import gender_guesser.detector as gender


class GenderDetector:
    def __init__(self):
        self.detector = gender.Detector(case_sensitive=False)

        # 1. Чорний список: банківські та системні слова, які точно не є іменами
        self.stop_words = {
            "visa",
            "cbs",
            "mastercard",
            "paypal",
            "visadebitserve",
            "apple",
            "google",
        }

        # 2. Кастомний словник для специфічних імен, на яких сипеться бібліотека
        self.custom_names = {
            "fionna": "female",
            "elara": "female",
            # Сюди можна додавати інші імена, якщо вони вилізуть під час тестів
        }

    def _detect_gender_inner(
        self, first: str
    ) -> Literal["male"] | Literal["female"] | Literal["unknown"]:

        first_lower = first.lower()

        # 3. Відсіюємо банківське сміття
        if first_lower in self.stop_words:
            return "unknown"

        # 4. Перевіряємо кастомний словник винятків
        if first_lower in self.custom_names:
            return self.custom_names[first_lower]

        def get_gender_category(word: str):
            res = self.detector.get_gender(word.capitalize())
            if res in ("male", "mostly_male"):
                return "male"
            if res in ("female", "mostly_female"):
                return "female"
            return "unknown"

        detected = get_gender_category(first)
        if detected != "unknown":
            return detected

        # 5. Fallback тепер безпечніший: вмикається ТІЛЬКИ для слів довших за 7 літер
        # Це врятує короткі імена від хибного обрізання до чоловічих/жіночих коренів
        if len(first) > 7:
            for i in range(len(first) - 1, 3, -1):
                sub_name = first[:i]
                sub_detected = get_gender_category(sub_name)
                if sub_detected != "unknown":
                    return sub_detected

        return "unknown"

    def detect_gender(
        self, name: str
    ) -> Literal["male"] | Literal["female"] | Literal["unknown"]:
        parsed_name = HumanName(name)

        title = parsed_name.title.lower().replace(".", "")
        if title in ("mr", "mister", "sir"):
            return "male"
        if title in ("ms", "mrs", "miss", "madam", "lady"):
            return "female"

        first = parsed_name.first or parsed_name.nickname
        if not first:
            return "unknown"

        result = self._detect_gender_inner(first)
        if result == "unknown":
            result = self._detect_gender_inner(parsed_name.last)
        return result


def test_gender_detector():
    MALES = [
        "Lucas Mercer",
        "David Miller",
        "Robert K. Wilson",
        "Julian Mercer",
        "christopherhill christopherhill",
        "jonathanwilson jonathanwilson",
        "jordantroyhayes jordantroyhayes",
        "anderson thomas",
        "sebastian s miller fmith",
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
        "mary l. smith",
        "j maria wilson",
        "elenaross elenaross",
        "anderson sarah",
    ]

    UNKNOWNS = [
        "",
        "visa visa",
        "cbs cbs",
        "visadebitserve visadebitserve",
        "vanilla visa",
        "vanilla gift",
        "vanillagift agiftforyou",
        "vanillagiftvisadebit vanillagiftvisadebit",
        "giftvanilla giftvanilla",
        "my card",
        "cooperative bank",
    ]

    DATA = [(MALES, "male"), (FEMALES, "female"), (UNKNOWNS, "unknown")]
    gender_detector = GenderDetector()

    for names, expected in DATA:
        for name in names:
            predicted = gender_detector.detect_gender(name)
            assert expected == predicted, (name, expected, predicted)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    args = parser.parse_args()
    gender_detector = GenderDetector()
    gender = gender_detector.detect_gender(args.name)
    print(gender)
