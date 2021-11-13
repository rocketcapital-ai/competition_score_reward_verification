from unittest import TestCase, main
from lib.scoring import *
import scipy.stats.mstats as mstats

CHALLENGE_1 = 1
CHALLENGE_4 = 4
CHALLENGE_5 = 5
CHALLENGE_6 = 6


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

    def test_compute_challenge_error(self):
        # maximum RMSE with base values in [0, 1] is 1
        self.assertEqual(dec(1), compute_challenge_error(CHALLENGE_1, [dec(0), dec(1), dec(0), dec(1)],
                                                         [dec(1), dec(0), dec(1), dec(0)]))

        # intermediate RMSE just for sake of completeness
        self.assertEqual(dec(np.sqrt(0.5)), compute_challenge_error(CHALLENGE_1, [dec(0), dec(1), dec(0), dec(1)],
                                                                    [dec(0), dec(0), dec(1), dec(1)]))

        # minimum RMSE is 0 when series are identical
        self.assertEqual(dec(0), compute_challenge_error(CHALLENGE_1, [dec(0), dec(1), dec(0), dec(1)],
                                                         [dec(0), dec(1), dec(0), dec(1)]))

    def test_compute_challenge_scores(self):

        # if there are no participants the challenge score list is empty
        self.assertEqual(compute_challenge_scores(CHALLENGE_1, []), [])

        # simple case where errors are all different
        errors = [0, 0.8, 0.2, 1]
        ranks = [1, 3, 2, 4]
        scores = [1 - (r - 1) / (len(ranks) - 1) for r in ranks]
        for x, y in zip(compute_challenge_scores(CHALLENGE_1, errors), scores):
            self.assertAlmostEqual(x, y)

        # less simple case some errors are equal
        errors = [0, 0.8, 0.2, 0.8, 1]
        ranks = [1, 3.5, 2, 3.5, 5]
        scores = [1 - (r - 1) / (len(ranks) - 1) for r in ranks]
        for x, y in zip(compute_challenge_scores(CHALLENGE_1, errors), scores):
            self.assertAlmostEqual(x, y)

        # simple case with some NaNs
        errors = [0, 0.8, np.nan, 0.2, 1, np.nan]
        ranks = mstats.rankdata(np.ma.masked_invalid(errors))
        ranks[ranks == 0] = np.nan
        scores = [1 - (r - 1) / 3 for r in ranks]
        for x, y in zip(compute_challenge_scores(CHALLENGE_1, errors), scores):
            if np.isnan(x):
                self.assertTrue(np.isnan(y))
            else:
                self.assertAlmostEqual(x, y)

    def test_compute_competition_score(self):

        # competition score with no submissions is NaN
        self.assertTrue(math.isnan(compute_competition_score(CHALLENGE_1, [])))

        # competition score with only NaN submissions is NaN
        self.assertTrue(
            math.isnan(compute_competition_score(CHALLENGE_1, [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])))

        # competition score with all identical submissions is the last value
        self.assertAlmostEqual(compute_competition_score(CHALLENGE_1, [0.6, 0.6, 0.6, 0.6, 0.6, 0.6]), 0.6)

        # competition score with 4 identical submissions a after a different value in the past is the last value
        self.assertAlmostEqual(compute_competition_score(CHALLENGE_1, [0.2, 0.2, 0.6, 0.6, 0.6, 0.6]), 0.6)

        # competition score with just 4 identical submissions is the last value
        self.assertAlmostEqual(compute_competition_score(CHALLENGE_1, [0.6, 0.6, 0.6, 0.6]), 0.6)

        # competition score with 3 identical submissions has a penalty of SKIP_PENALTY * 1/3
        self.assertAlmostEqual(compute_competition_score(CHALLENGE_1, [0.6, np.nan, 0.6, 0.6]),
                               0.6 - Scorer1.SKIP_PENALTY / 3)
        self.assertAlmostEqual(compute_competition_score(CHALLENGE_1, [0.6, 0.6, 0.6]),
                               0.6 - Scorer1.SKIP_PENALTY / 3)

        # competition score with 2 identical submissions has a penalty of SKIP_PENALTY * 2/3
        self.assertAlmostEqual(compute_competition_score(CHALLENGE_1, [0.6, np.nan, 0.6, np.nan]),
                               0.6 - Scorer1.SKIP_PENALTY * 2 / 3)
        self.assertAlmostEqual(compute_competition_score(CHALLENGE_1, [0.6, 0.6]),
                               0.6 - Scorer1.SKIP_PENALTY * 2 / 3)

        # competition score with 1 submissions has a penalty of SKIP_PENALTY
        self.assertAlmostEqual(compute_competition_score(CHALLENGE_1, [np.nan, np.nan, 0.6, np.nan]),
                               0.6 - Scorer1.SKIP_PENALTY)

        # competition with 4 scores with maximum b factor
        self.assertAlmostEqual(compute_competition_score(CHALLENGE_1, [0, 1, 0, 1]),
                               0.5 - Scorer1.STDDEV_PENALTY)

        # competition score with 4 submissions is average minus STDDEV_PENALTY * stddev
        for _ in range(100):
            challenge_scores = np.random.random_sample(10)
            a = challenge_scores[-4:].mean()
            b = challenge_scores[-4:].std()
            competition_score = a - Scorer1.STDDEV_PENALTY * 2 * b
            self.assertAlmostEqual(compute_competition_score(CHALLENGE_1, challenge_scores), competition_score)

    def test_compute_challenge_rewards(self):

        # [0.5, 0.25, 0.75, 1, 0] should give [16.6666666666, 0, 33.3333333333, 50, 0]
        challenge_pool = Decimal("100");
        challenge_scores = [0.5, 0.25, 0.75, 1, 0]
        expected_rewards = [dec("16.6666666666"), dec(0), dec("33.3333333333"), dec("50"), dec(0)]
        challenge_rewards = compute_challenge_rewards(CHALLENGE_1, challenge_scores, challenge_pool)
        self.assertEqual(expected_rewards, challenge_rewards)

        # same with two nans
        challenge_scores = [0.5, np.nan, 0.25, 0.75, 1, 0, np.nan]
        expected_rewards = [dec("16.6666666666"), dec(0), dec(0), dec("33.3333333333"), dec("50"), dec(0), dec(0)]
        challenge_rewards = compute_challenge_rewards(CHALLENGE_1, challenge_scores, challenge_pool)
        self.assertEqual(expected_rewards, challenge_rewards)

        # special case for challenge 5
        expected_rewards = [dec("37.1428571428")] * 28
        challenge_rewards = compute_challenge_rewards(CHALLENGE_5, [], dec(1040))
        self.assert_almost_equals_00000001(expected_rewards, challenge_rewards)

    def test_compute_competition_rewards(self):

        # [0.33, 0.12, 0.6, 0.7, 0.1] should give [0, 0, 66.6666666666, 133.3333333333, 0]
        competition_pool = Decimal("200");
        competition_scores = [0.33, 0.12, 0.6, 0.7, 0.1]
        expected_rewards = [dec(0), dec(0), dec("66.6666666666"), dec("133.3333333333"), dec(0)]
        competition_rewards = compute_competition_rewards(CHALLENGE_1, competition_scores, competition_pool)
        self.assertEqual(expected_rewards, competition_rewards)

        # same with two nans
        competition_scores = [np.nan, 0.33, 0.12, 0.6, np.nan, 0.7, 0.1]
        expected_rewards = [dec(0), dec(0), dec(0), dec("66.6666666666"), dec(0), dec("133.3333333333"), dec(0)]
        competition_rewards = compute_competition_rewards(CHALLENGE_1, competition_scores, competition_pool)
        self.assertEqual(expected_rewards, competition_rewards)

        # special case for challenge 5
        expected_rewards = [dec("111.4285714286")] * 28
        competition_rewards = compute_competition_rewards(CHALLENGE_5, [], dec(3120))
        self.assert_almost_equals_00000001(expected_rewards, competition_rewards)

    def test_compute_stake_rewards(self):

        # [23, 65, 34, 87, 12] should give [10.407239819, 29.4117647058, 15.3846153846, 39.3665158371, 5.4298642533]
        stakes = [dec(23), dec(65), dec(34), dec(87), dec(12)]
        stake_pool = dec(100)
        expected_rewards = [dec("10.407239819"), dec("29.4117647058"), dec("15.3846153846"),
                            dec("39.3665158371"), dec("5.4298642533")]
        stake_rewards = compute_stake_rewards(CHALLENGE_1, stakes, stake_pool)
        self.assertEqual(expected_rewards, stake_rewards)

    def test_compute_challenge_pool(self):
        self.assertEqual(dec(0), compute_challenge_pool(CHALLENGE_1, 0))
        self.assertEqual(dec(1560), compute_challenge_pool(CHALLENGE_1, 39))
        self.assertEqual(dec(40000), compute_challenge_pool(CHALLENGE_1, 1000))
        self.assertEqual(dec(40000), compute_challenge_pool(CHALLENGE_1, 1001))
        self.assertEqual(dec(40000), compute_challenge_pool(CHALLENGE_1, 2000))

    def test_compute_competition_pool(self):
        self.assertEqual(dec(0), compute_competition_pool(CHALLENGE_1, 0))
        self.assertEqual(dec(4680), compute_competition_pool(CHALLENGE_1, 39))
        self.assertEqual(dec(120000), compute_competition_pool(CHALLENGE_1, 1000))
        self.assertEqual(dec(120000), compute_competition_pool(CHALLENGE_1, 1001))
        self.assertEqual(dec(120000), compute_competition_pool(CHALLENGE_1, 2000))

    def test_compute_stake_pool(self):
        self.assertEqual(dec(0), compute_stake_pool(CHALLENGE_1, 0, 0))
        self.assertEqual(dec(1960), compute_stake_pool(CHALLENGE_1, 0, 49))
        self.assertEqual(dec(40000), compute_stake_pool(CHALLENGE_1, 0, 1000))
        self.assertEqual(dec(40000), compute_stake_pool(CHALLENGE_1, 0, 1001))
        self.assertEqual(dec(40000), compute_stake_pool(CHALLENGE_1, 0, 2000))

        self.assertEqual(dec(0), compute_stake_pool(CHALLENGE_4, 0, 0))
        self.assertEqual(dec(1960), compute_stake_pool(CHALLENGE_4, 0, 49))
        self.assertEqual(dec(40000), compute_stake_pool(CHALLENGE_4, 0, 1000))
        self.assertEqual(dec(40000), compute_stake_pool(CHALLENGE_4, 0, 1001))
        self.assertEqual(dec(40000), compute_stake_pool(CHALLENGE_4, 0, 2000))

        self.assertEqual(dec(0), compute_stake_pool(CHALLENGE_6, 0, 1000))
        self.assertEqual(dec(1560), compute_stake_pool(CHALLENGE_6, 39, 1000))
        self.assertEqual(dec(20000), compute_stake_pool(CHALLENGE_6, 500, 1000))
        self.assertEqual(dec(40000), compute_stake_pool(CHALLENGE_6, 1000, 1000))
        self.assertEqual(dec(40000), compute_stake_pool(CHALLENGE_6, 1001, 1000))
        self.assertEqual(dec(40000), compute_stake_pool(CHALLENGE_6, 2000, 1000))

    def test_compute_pool_surplus(self):
        self.assertEqual(dec(200000), compute_pool_surplus(CHALLENGE_1, 0, 0))
        self.assertEqual(dec(191800), compute_pool_surplus(CHALLENGE_1, 39, 49))
        self.assertEqual(dec(0), compute_pool_surplus(CHALLENGE_1, 1000, 1000))

    # two lists differs at most 1e-10
    def assert_almost_equals_00000001(self, xs: [Decimal], ys: [Decimal]) -> None:
        for x, y in zip(xs, ys):
            self.assertTrue(abs(x - y) <= dec("0.0000000001"))
        pass


if __name__ == "__main__":
    main()
