"""
 Copyright (C) <year(s)> Intel Corporation

 SPDX-License-Identifier: Apache-2.0

"""
import unittest

import convAssist.word_sentence_predictor
import convAssist.combiner


class TestMeritocracyCombiner(unittest.TestCase):
    def setUp(self):
        self.combiner = convAssist.combiner.MeritocracyCombiner()

    def _create_prediction(self):
        prediction = convAssist.word_sentence_predictor.Prediction()
        prediction.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test", 0.3))
        prediction.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test2", 0.3))
        prediction.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test", 0.1))
        prediction.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test3", 0.2))
        return prediction

    def _create_prediction2(self):
        prediction = convAssist.word_sentence_predictor.Prediction()
        prediction.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test2", 0.3))
        prediction.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test", 0.1))
        prediction.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test3", 0.2))
        return prediction

    def test_filter(self):
        result = self.combiner.filter(self._create_prediction())

        correct = convAssist.word_sentence_predictor.Prediction()
        correct.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test3", 0.2))
        correct.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test2", 0.3))
        correct.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test", 0.4))

        assert result == correct

    def test_combine(self):
        predictions = [self._create_prediction2()]
        prediction2 = self._create_prediction2()
        prediction2.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test4", 0.1))
        predictions.append(prediction2)
        result = self.combiner.combine(predictions)

        correct = convAssist.word_sentence_predictor.Prediction()
        correct.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test3", 0.4))
        correct.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test2", 0.6))
        correct.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test4", 0.1))
        correct.add_suggestion(convAssist.word_sentence_predictor.Suggestion("Test", 0.2))

        assert result == correct
