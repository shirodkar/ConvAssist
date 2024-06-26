# Copyright (C) 2023 Intel Corporation


# SPDX-License-Identifier: Apache-2.0

"""
Base class for callbacks.

"""

from __future__ import absolute_import, unicode_literals


class Callback(object):
    """
    Base class for callbacks.

    """

    def __init__(self):
        self.stream = ""
        self.empty = ""

    def past_stream(self):
        return self.stream

    def future_stream(self):
        return self.empty

    def update(self, character):
        if character == "\b" and len(self.stream) > 0:
            self.stream[:-1]
        else:
            self.stream += character
