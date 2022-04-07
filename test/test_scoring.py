from unittest import TestCase, main
from lib.scoring import *
import scipy.stats.mstats as mstats


class Test(TestCase):

    def test_validate_prediction(self):
        assets = ["AAPL", "GOOG"]

        # empty prediction is not valid
        self.assertFalse(validate_prediction(assets, []))

        # incomplete prediction is not valid
        prediciton = [("AAPL", Decimal("0.1"))]
        self.assertFalse(validate_prediction(assets, prediciton))

        # redundant prediction is not valid
        prediction = [("AAPL", Decimal("0.1")), ("GOOG", Decimal("0.2")), ("AAPL", Decimal("0.3"))]
        self.assertFalse(validate_prediction(assets, prediction))

        # prediction with extra entry is not valid
        prediction = [("AAPL", Decimal("0.1")), ("GOOG", Decimal("0.2")), ("MSFT", Decimal("0.3"))]
        self.assertFalse(validate_prediction(assets, prediction))

        # prediction with NaN entry is not valid
        prediction = [("AAPL", Decimal("0.1")), ("GOOG", Decimal(math.nan))]
        self.assertFalse(validate_prediction(assets, prediction))

        # this prediction is valid
        prediction = [("AAPL", Decimal("0.1")), ("GOOG", Decimal("0.2"))]
        self.assertTrue(validate_prediction(assets, prediction))

    def test_validate_prediction_with_repeated_assets(self):

        # redundant prediction is not valid
        prediction = [("AAPL", Decimal("0.1")), ("GOOG", Decimal("0.2")), ("AAPL", Decimal("0.3"))]
        self.assertFalse(validate_prediction(["AAPL", "GOOG"], prediction))

        # this prediction is not valid due to repeated assets
        prediction = [("AAPL", Decimal("0.1")), ("GOOG", Decimal("0.2"))]
        self.assertFalse(validate_prediction(["AAPL", "GOOG", "AAPL"], prediction))

    def test_compute_RMSE_raw_score(self):
        # maximum RMSE with base values in [0, 1] is 1
        self.assertEqual(dec(1), compute_raw_score(1, [dec(0), dec(1), dec(0), dec(1)],
                                                      [dec(1), dec(0), dec(1), dec(0)]))

        # intermediate RMSE just for sake of completeness
        self.assertEqual(dec(np.sqrt(0.5)), compute_raw_score(1, [dec(0), dec(1), dec(0), dec(1)],
                                                                 [dec(0), dec(0), dec(1), dec(1)]))

        # minimum RMSE is 0 when series are identical
        self.assertEqual(dec(0), compute_raw_score(1, [dec(0), dec(1), dec(0), dec(1)],
                                                      [dec(0), dec(1), dec(0), dec(1)]))

        # maximum RMSE still 1 at challenge 26
        self.assertEqual(dec(1), compute_raw_score(26, [dec(0), dec(1), dec(0), dec(1)],
                                                       [dec(1), dec(0), dec(1), dec(0)]))


    def test_compute_Spearman_raw_score(self):
        # minimum spearman correlation
        self.assertEqual(dec(-1), compute_raw_score(27, [dec(1), dec(2), dec(3), dec(4)],
                                                        [dec(4), dec(3), dec(2), dec(1)]))
        # zero spearman correlation
        self.assertEqual(dec(0), compute_raw_score(27, [dec(1), dec(2), dec(3), dec(4)],
                                                       [dec(3), dec(1), dec(4), dec(2)]))
        # maximum spearman correlation
        self.assertEqual(dec(1), compute_raw_score(27, [dec(1), dec(2), dec(3), dec(4)],
                                                       [dec(1), dec(2), dec(3), dec(4)]))
        # minimum spearman correlation
        self.assertEqual(dec(-1), compute_raw_score(27, [dec(110), dec(120), dec(130), dec(140)],
                                                    [dec(140), dec(130), dec(120), dec(110)]))
        # zero spearman correlation
        self.assertEqual(dec(0), compute_raw_score(27, [dec(110), dec(120), dec(130), dec(140)],
                                                       [dec(130), dec(110), dec(140), dec(120)]))
        # maximum spearman correlation
        self.assertEqual(dec(1), compute_raw_score(27, [dec(110), dec(120), dec(130), dec(140)],
                                                       [dec(110), dec(120), dec(130), dec(140)]))

    def test_compute_challenge_scores(self):

        # if there are no participants the challenge score list is empty
        self.assertEqual(compute_challenge_scores(1, []), [])

        # simple case where errors are all different
        errors = [0, 0.8, 0.2, 1]
        ranks = [1, 3, 2, 4]
        scores = [1 - (r - 1) / (len(ranks) - 1) for r in ranks]
        for x, y in zip(compute_challenge_scores(1, errors), scores):
            self.assertAlmostEqual(x, y)

        # less simple case some errors are equal
        errors = [0, 0.8, 0.2, 0.8, 1]
        ranks = [1, 3.5, 2, 3.5, 5]
        scores = [1 - (r - 1) / (len(ranks) - 1) for r in ranks]
        for x, y in zip(compute_challenge_scores(1, errors), scores):
            self.assertAlmostEqual(x, y)

        # simple case with some NaNs
        errors = [0, 0.8, np.nan, 0.2, 1, np.nan]
        ranks = mstats.rankdata(np.ma.masked_invalid(errors))
        ranks[ranks == 0] = np.nan
        scores = [1 - (r - 1) / 3 for r in ranks]
        for x, y in zip(compute_challenge_scores(1, errors), scores):
            if np.isnan(x):
                self.assertTrue(np.isnan(y))
            else:
                self.assertAlmostEqual(x, y)

        # simple case where errors are all different still valid at challenge 26
        errors = [0, 0.8, 0.2, 1]
        ranks = [1, 3, 2, 4]
        scores = [1 - (r - 1) / (len(ranks) - 1) for r in ranks]
        for x, y in zip(compute_challenge_scores(26, errors), scores):
            self.assertAlmostEqual(x, y)

        # case with spearman correlation, with reversed order, valid from challenge 27
        spearman_corrs = [0, 0.8, 0.2, 1]
        ranks = [1, 3, 2, 4]
        scores = [(r - 1) / (len(ranks) - 1) for r in ranks]
        for x, y in zip(compute_challenge_scores(27, spearman_corrs), scores):
            self.assertAlmostEqual(x, y)

    def test_compute_competition_score(self):
        self._test_compute_competition_score_up_to_17(1)
        self._test_compute_competition_score_up_to_17(17)
        self._test_compute_competition_score_from_18(18)

    def _test_compute_competition_score_up_to_17(self, challenge_number):
        scorer = Scorer.get(challenge_number)

        # competition score with no submissions is NaN
        self.assertTrue(math.isnan(compute_competition_score(challenge_number, [])))

        # competition score with only NaN submissions is NaN
        self.assertTrue(
            math.isnan(compute_competition_score(challenge_number, [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])))

        # competition score with all identical submissions is the last value
        self.assertAlmostEqual(compute_competition_score(challenge_number, [0.6, 0.6, 0.6, 0.6, 0.6, 0.6]), 0.6)

        # competition score with 4 identical submissions a after a different value in the past is the last value
        self.assertAlmostEqual(compute_competition_score(challenge_number, [0.2, 0.2, 0.6, 0.6, 0.6, 0.6]), 0.6)

        # competition score with 4 identical submissions is the last value
        self.assertAlmostEqual(compute_competition_score(challenge_number, [0.6, 0.6, 0.6, 0.6]), 0.6)

        # competition score with just 3 identical submissions has a penalty of SKIP_PENALTY * 1/3
        self.assertAlmostEqual(compute_competition_score(challenge_number, [0.6, np.nan, 0.6, 0.6]),
                               0.6 - scorer.get_skip_penalty() / 3)
        self.assertAlmostEqual(compute_competition_score(challenge_number, [0.6, 0.6, 0.6]),
                               0.6 - scorer.get_skip_penalty() / 3)

        # competition score with just 2 identical submissions has a penalty of SKIP_PENALTY * 2/3
        self.assertAlmostEqual(compute_competition_score(challenge_number, [0.6, np.nan, 0.6, np.nan]),
                               0.6 - scorer.get_skip_penalty() * 2 / 3)
        self.assertAlmostEqual(compute_competition_score(challenge_number, [0.6, 0.6]),
                               0.6 - scorer.get_skip_penalty() * 2 / 3)

        # competition score with just 1 submissions has a penalty of SKIP_PENALTY
        self.assertAlmostEqual(compute_competition_score(challenge_number, [np.nan, np.nan, 0.6, np.nan]),
                               0.6 - scorer.get_skip_penalty())
        self.assertAlmostEqual(compute_competition_score(challenge_number, [0.6]),
                               0.6 - scorer.get_skip_penalty())

        # competition with 4 scores with maximum b factor
        self.assertAlmostEqual(compute_competition_score(challenge_number, [0, 1, 0, 1]),
                               0.5 - scorer.get_std_dev_penalty())

        # competition score with 4 submissions is average minus STDDEV_PENALTY * stddev
        for _ in range(100):
            challenge_scores = np.random.random_sample(10)
            a = challenge_scores[-4:].mean()
            b = challenge_scores[-4:].std()
            competition_score = a - scorer.get_std_dev_penalty() * 2 * b
            self.assertAlmostEqual(compute_competition_score(challenge_number, challenge_scores), competition_score)

    # test new parameters: stddev = 0.2, skip = 0.5, window_size = 8
    # note: this could be merged with the _test_compute_competition_score()
    # but it is left duplicated for sake of simplicity
    def _test_compute_competition_score_from_18(self, challenge_number):
        scorer = Scorer.get(challenge_number)

        # competition score with no submissions is NaN
        self.assertTrue(math.isnan(compute_competition_score(challenge_number, [])))

        # competition score with 12 NaN submissions is NaN
        self.assertTrue(
            math.isnan(compute_competition_score(challenge_number, [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                                                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])))

        # competition score with 12 identical submissions is the last value
        self.assertAlmostEqual(compute_competition_score(challenge_number, [0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                                                                            0.6, 0.6, 0.6, 0.6, 0.6, 0.6]), 0.6)

        # competition score with 8 identical submissions after a different value in the past is the last value
        self.assertAlmostEqual(compute_competition_score(challenge_number, [0.2, 0.2, 0.2, 0.2, 0.6, 0.6,
                                                                            0.6, 0.6, 0.6, 0.6, 0.6, 0.6]), 0.6)

        # competition score with just 8 identical submissions is the last value
        self.assertAlmostEqual(compute_competition_score(challenge_number, [0.6, 0.6, 0.6, 0.6,
                                                                            0.6, 0.6, 0.6, 0.6]), 0.6)

        # competition score with just 7 identical submissions has a penalty of SKIP_PENALTY * 1/7
        self.assertAlmostEqual(compute_competition_score(challenge_number, [0.6, np.nan, 0.6, 0.6,
                                                                            0.6, 0.6, 0.6, 0.6]),
                               0.6 - scorer.get_skip_penalty() / 7)
        self.assertAlmostEqual(compute_competition_score(challenge_number, [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]),
                               0.6 - scorer.get_skip_penalty() / 7)

        # competition score with just 4 identical submissions has a penalty of SKIP_PENALTY * 4/7
        self.assertAlmostEqual(compute_competition_score(challenge_number, [0.6, np.nan, 0.6, np.nan,
                                                                            0.6, np.nan, 0.6, np.nan]),
                               0.6 - scorer.get_skip_penalty() * 4 / 7)
        self.assertAlmostEqual(compute_competition_score(challenge_number, [0.6, 0.6, 0.6, 0.6]),
                               0.6 - scorer.get_skip_penalty() * 4 / 7)

        # competition score with just 1 submission has a penalty of SKIP_PENALTY
        self.assertAlmostEqual(compute_competition_score(challenge_number, [np.nan, np.nan, np.nan, 0.6,
                                                                            np.nan, np.nan, np.nan, np.nan]),
                               0.6 - scorer.get_skip_penalty())
        self.assertAlmostEqual(compute_competition_score(challenge_number, [0.6]),
                               0.6 - scorer.get_skip_penalty())

        # competition with 8 scores with maximum b factor
        self.assertAlmostEqual(compute_competition_score(challenge_number, [0, 1, 0, 1, 0, 1, 0, 1]),
                               0.5 - scorer.get_std_dev_penalty())

        # competition score with 4 submissions is average minus STDDEV_PENALTY * stddev
        for _ in range(100):
            challenge_scores = np.random.random_sample(10)
            a = challenge_scores[-8:].mean()
            b = challenge_scores[-8:].std()
            competition_score = a - scorer.get_std_dev_penalty() * 2 * b
            self.assertAlmostEqual(compute_competition_score(challenge_number, challenge_scores), competition_score)

    def test_compute_challenge_rewards(self):

        # [0.5, 0.25, 0.75, 1, 0] should give [16.6666666666, 0, 33.3333333333, 50, 0]
        challenge_pool = Decimal("100")
        challenge_scores = [0.5, 0.25, 0.75, 1, 0]
        expected_rewards = [dec("16.6666666666"), dec(0), dec("33.3333333333"), dec("50"), dec(0)]
        challenge_rewards = compute_challenge_rewards(1, challenge_scores, challenge_pool)
        self.assertEqual(expected_rewards, challenge_rewards)

        # same with two nans
        challenge_scores = [0.5, np.nan, 0.25, 0.75, 1, 0, np.nan]
        expected_rewards = [dec("16.6666666666"), dec(0), dec(0), dec("33.3333333333"), dec("50"), dec(0), dec(0)]
        challenge_rewards = compute_challenge_rewards(1, challenge_scores, challenge_pool)
        self.assertEqual(expected_rewards, challenge_rewards)

        # special case for challenge 5
        expected_rewards = [dec("37.1428571428")] * 28
        challenge_rewards = compute_challenge_rewards(5, [], dec(1040))
        self.assert_almost_equals_00000001(expected_rewards, challenge_rewards)

    def test_compute_competition_rewards(self):

        # [0.33, 0.12, 0.6, 0.7, 0.1] should give [0, 0, 66.6666666666, 133.3333333333, 0]
        competition_pool = Decimal("200")
        competition_scores = [0.33, 0.12, 0.6, 0.7, 0.1]
        challenge_scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        expected_rewards = [dec(0), dec(0), dec("66.6666666666"), dec("133.3333333333"), dec(0)]
        competition_rewards = compute_competition_rewards(1, competition_scores,
                                                          challenge_scores, competition_pool)
        self.assertEqual(expected_rewards, competition_rewards)

        # same with two nans
        competition_scores = [np.nan, 0.33, 0.12, 0.6, np.nan, 0.7, 0.1]
        challenge_scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        expected_rewards = [dec(0), dec(0), dec(0), dec("66.6666666666"), dec(0), dec("133.3333333333"), dec(0)]
        competition_rewards = compute_competition_rewards(1, competition_scores,
                                                          challenge_scores, competition_pool)
        self.assertEqual(expected_rewards, competition_rewards)

        # special case for challenge 5
        expected_rewards = [dec("111.4285714286")] * 28
        competition_rewards = compute_competition_rewards(5, [], [], dec(3120))
        self.assert_almost_equals_00000001(expected_rewards, competition_rewards)

    def test_compute_competition_rewards_from_18(self):

        # [*0.5,   0.33, 0.12, 0.6, *0.5,   0.7, 0.1] with two nans in the challenge score
        # [np.nan, 0.5,  0.5,  0.5, np.nan, 0.5, 0.5]
        # should give the same rewards as [np.nan, 0.33, 0.12, 0.6, np.nan, 0.7, 0.1]
        competition_pool = Decimal("200")
        competition_scores = [0.5,    0.33, 0.12, 0.6, 0.5,    0.7, 0.1]
        challenge_scores =   [np.nan, 0.5,  0.5,  0.5, np.nan, 0.5, 0.5]
        expected_rewards = [dec(0), dec(0), dec(0), dec("66.6666666666"), dec(0), dec("133.3333333333"), dec(0)]
        competition_rewards = compute_competition_rewards(18, competition_scores,
                                                          challenge_scores, competition_pool)
        self.assertEqual(expected_rewards, competition_rewards)

    def test_compute_stake_rewards(self):

        # [23, 65, 34, 87, 12] should give [10.407239819, 29.4117647058, 15.3846153846, 39.3665158371, 5.4298642533]
        stakes = [dec(23), dec(65), dec(34), dec(87), dec(12)]
        stake_pool = dec(100)
        expected_rewards = [dec("10.407239819"), dec("29.4117647058"), dec("15.3846153846"),
                            dec("39.3665158371"), dec("5.4298642533")]
        stake_rewards = compute_stake_rewards(1, stakes, stake_pool)
        self.assertEqual(expected_rewards, stake_rewards)

    def test_compute_challenge_pool(self):
        self.assertEqual(dec(0), compute_challenge_pool(1, 0))
        self.assertEqual(dec(1560), compute_challenge_pool(1, 39))
        self.assertEqual(dec(40000), compute_challenge_pool(1, 1000))
        self.assertEqual(dec(40000), compute_challenge_pool(1, 1001))
        self.assertEqual(dec(40000), compute_challenge_pool(1, 2000))

    def test_compute_competition_pool(self):
        self.assertEqual(dec(0), compute_competition_pool(1, 0))
        self.assertEqual(dec(4680), compute_competition_pool(1, 39))
        self.assertEqual(dec(120000), compute_competition_pool(1, 1000))
        self.assertEqual(dec(120000), compute_competition_pool(1, 1001))
        self.assertEqual(dec(120000), compute_competition_pool(1, 2000))

    def test_compute_stake_pool(self):
        self.assertEqual(dec(0), compute_stake_pool(1, 0, 0))
        self.assertEqual(dec(1960), compute_stake_pool(1, 0, 49))
        self.assertEqual(dec(40000), compute_stake_pool(1, 0, 1000))
        self.assertEqual(dec(40000), compute_stake_pool(1, 0, 1001))
        self.assertEqual(dec(40000), compute_stake_pool(1, 0, 2000))

        self.assertEqual(dec(0), compute_stake_pool(4, 0, 0))
        self.assertEqual(dec(1960), compute_stake_pool(4, 0, 49))
        self.assertEqual(dec(40000), compute_stake_pool(4, 0, 1000))
        self.assertEqual(dec(40000), compute_stake_pool(4, 0, 1001))
        self.assertEqual(dec(40000), compute_stake_pool(4, 0, 2000))

        self.assertEqual(dec(0), compute_stake_pool(6, 0, 1000))
        self.assertEqual(dec(1560), compute_stake_pool(6, 39, 1000))
        self.assertEqual(dec(20000), compute_stake_pool(6, 500, 1000))
        self.assertEqual(dec(40000), compute_stake_pool(6, 1000, 1000))
        self.assertEqual(dec(40000), compute_stake_pool(6, 1001, 1000))
        self.assertEqual(dec(40000), compute_stake_pool(6, 2000, 1000))

    def test_compute_pool_surplus(self):
        self.assertEqual(dec(200000), compute_pool_surplus(1, 0, 0))
        self.assertEqual(dec(191800), compute_pool_surplus(1, 39, 49))
        self.assertEqual(dec(0), compute_pool_surplus(1, 1000, 1000))

    # two lists differs at most 1e-10
    def assert_almost_equals_00000001(self, xs: [Decimal], ys: [Decimal]) -> None:
        for x, y in zip(xs, ys):
            self.assertTrue(abs(x - y) <= dec("0.0000000001"))
        pass


if __name__ == "__main__":
    main()
