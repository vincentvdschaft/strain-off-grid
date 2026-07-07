import math


def scientific_notation(number, decimals=1):
    """
    Convert a number to scientific notation with a fixed number of decimals of form
    $a \times 10^{b}$.
    """
    if number == 0:
        return "0"
    exponent = int(math.log10(abs(number)))
    mantissa = number / 10**exponent
    return f"{mantissa:.{decimals}f} \\times 10^{{{exponent}}}"


def scientific_notation_e(number, decimals=1):
    """
    Convert a number to scientific notation with a fixed number of decimals of form
    $a\text{e}b$.
    """
    if number == 0:
        return "0"
    exponent = int(math.log10(abs(number)))
    mantissa = number / 10**exponent
    return "".join([f"{mantissa:.{decimals}f}", r"\text{e}", f"{exponent}"])


if __name__ == "__main__":
    print(scientific_notation(0))
    print(scientific_notation(1))
    print(scientific_notation(10231))
    print(scientific_notation_e(0))
    print(scientific_notation_e(1))
    print(scientific_notation_e(10231))
