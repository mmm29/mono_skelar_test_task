# name_matcher.py

from nameparser import HumanName

def _match_part_or_initial(part1: str, part2: str) -> bool:
    """
    Допоміжна функція: перевіряє точний збіг або збіг за ініціалом.
    Наприклад: 'david' та 'd' -> True, 'david' та 'm' -> False
    """
    if not part1 or not part2:
        return True  # Якщо в одного імені цієї частини немає, вважаємо, що конфлікту немає
        
    part1, part2 = part1.lower().replace('.', ''), part2.lower().replace('.', '')
    
    if part1 == part2:
        return True
        
    # Перевірка на ініціал (одна літера)
    if len(part1) == 1 and part2.startswith(part1):
        return True
    if len(part2) == 1 and part1.startswith(part2):
        return True
        
    return False

def is_same_person(name_a: str, name_b: str) -> bool:
    """
    Порівнює два імені та повертає True, якщо це ймовірно одна людина.
    Враховує ініціали, титули та відсутні middle names.
    """
    # Якщо рядки абсолютно ідентичні (швидка перевірка)
    if name_a.strip().lower() == name_b.strip().lower():
        return True
        
    hn_a = HumanName(name_a)
    hn_b = HumanName(name_b)

    # 1. Прізвище (Last Name) має збігатися обов'язково
    if hn_a.last.lower() != hn_b.last.lower():
        return False

    # 2. Ім'я (First Name) - точний збіг або ініціал
    if not _match_part_or_initial(hn_a.first, hn_b.first):
        return False

    # 3. По батькові / Друге ім'я (Middle Name) - перевіряємо, тільки якщо воно є в обох
    if hn_a.middle and hn_b.middle:
        if not _match_part_or_initial(hn_a.middle, hn_b.middle):
            return False

    return True


# Блок для тестування (можна запустити цей файл окремо, щоб перевірити)
if __name__ == "__main__":
    # Тести з твого прикладу (мають повернути True)
    assert is_same_person("david p smith", "david smith") == True
    assert is_same_person("david smith", "d smith") == True
    assert is_same_person("d smith", "mr d smith") == True
    assert is_same_person("david p smith", "mr d smith") == True
    assert is_same_person("david", "david") == True
    
    # Тести на відмову (мають повернути False)
    assert is_same_person("david smith", "michael p scott") == False
    assert is_same_person("david smith", "davis smith") == False # Різні імена
    assert is_same_person("david p smith", "david m smith") == False # Різні middle names

    print("Всі тести успішно пройдені! Алгоритм працює ідеально.")
