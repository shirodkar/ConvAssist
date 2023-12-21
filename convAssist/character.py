# Copyright (C) <year(s)> Intel Corporation


# SPDX-License-Identifier: Apache-2.0

"""
Class to define character-related variables and functions in Pressagio. 
"""

import unicodedata
blankspaces = " \f\n\r\t\v  "

### Separators as defined in original pressagio codebase
# separators = '`~!@#$%^&*()_+=\\|]}[{";:/?.>,<¡¿†¨„“”«»।॥ו–—―´’‘‚י0123456789ः'

### removing apostrophe (') as a separator
separators =  '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'

### remove sentence endings as separators - ?!.
# separators =  '"#$%&()*+,-/:;<=>@[\]^_`{|}~'

def first_word_character(string):
    for i, ch in enumerate(string):
        if is_word_character(ch):
            return i

    return -1


def last_word_character(string):
    result = first_word_character(string[::-1])
    if result == -1:
        return -1
    return len(string) - result - 1


def is_word_character(char):
    # check for letter category
    if unicodedata.category(char)[0] == "L":
        return True
    return False
