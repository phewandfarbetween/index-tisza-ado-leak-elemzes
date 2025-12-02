import random
import sys

for line in sys.stdin.readlines():
    ocrified = ""
    for i, char in enumerate(line):
        if char == "ó" and random.random() < 0.5:
            char = "6"
        elif char == "í" and random.random() < 0.3:
            char = "i"
        elif char == "Í" and random.random() < 0.3:
            char = "I"
        elif char == "l" and random.random() < 0.02:
            char = "1"
        elif char == "ű" and random.random() < 0.3:
            char = "ü"
        elif char == "Ű" and random.random() < 0.3:
            char = "Ü"
        elif char == " " and i == 0 and random.random() < 0.3:
            char = "e "
        elif char == " " and random.random() < 0.03:
            char = "\n"
        elif char in "óöő" and random.random() < 0.5:
            char = "o"
        elif char in "ÓÖŐ" and random.random() < 0.5:
            char = "O"
        elif char in "úüű" and random.random() < 0.5:
            char = "u"
        elif char in "ÚÜŰ" and random.random() < 0.5:
            char = "U"
        ocrified += char
    print(ocrified)
