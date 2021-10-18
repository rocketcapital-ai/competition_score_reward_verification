import numpy as np
import lib.validation as val

from decimal import Decimal
from unittest import TestCase, main

from lib.rocket import get_rocket_competition


class Test(TestCase):

    def setUp(self):
        self.competition = get_rocket_competition()
        self.challenge = self.competition.get_challenge(1)

    def test_compute_error(self):
        self.assertAlmostEqual(0.28160952734193934, val.compute_error(self.challenge.get_participant("dbcc5a6c8b126e6be68b2bfb838d23ade8e7de57")), places=10)
        self.assertAlmostEqual(0.3168085074526588, val.compute_error(self.challenge.get_participant("7e3f69b71019b93123114375119d18df6a1f3a2a")), places=10)
        self.assertTrue(np.isnan(val.compute_error(self.challenge.get_participant("8355b0536a6a3b0116374b33dc3f8b7c344ca8bb"))))

    def test_compute_challenge_scores(self):
        expected_scores = {'16391fe372c2f5b2a3874e65187dd173e6552fe6': np.nan,
                           'dbcc5a6c8b126e6be68b2bfb838d23ade8e7de57': 1.0,
                           '889432c467b59475900845c1f414e0d3461d1459': 0.0666666666666666,
                           '7e3f69b71019b93123114375119d18df6a1f3a2a': 0.2,
                           '4c2a8ed7374912ee6c980b29fe5021aa94f4c543': 0.1333333333333333,
                           'b7ece21c962697a3194c0b6e4b81db54433055ab': 0.2666666666666666,
                           'a16a483a18f00a00b14b93438e8ad21f485b382f': 0.4666666666666667,
                           'fe3c3f351e6df846566e10119e4295ed69b2cd72': 0.6,
                           'f097ed117dd3930f1eeb91569a6a61b4c52744c8': 0.4,
                           '3e04d3e08ea54c34f721ac007107fe7a4bd957d5': 0.3333333333333333,
                           '1be48b2b7594fe7e47701bc7e7bb83beaf84ecc0': 0.5333333333333333,
                           'fb8bc3433016a64a3b8aa23c0275afe7c20feb53': 0.8666666666666667,
                           'eba5ba85f72e40a9f2b8782c2bd51ec31a5564da': 0.9333333333333332,
                           'e14867e41410bd7232a52b9400476c08b4e535b3': 0.7333333333333333,
                           '98e41d65e981a882afcb05d41939197da3465b6f': 0.8,
                           '8355b0536a6a3b0116374b33dc3f8b7c344ca8bb': np.nan,
                           '2547002b0b8bfd052111981ef314118b6b54f61c': 0.6666666666666666,
                           'bf72960a4051d7f0a3fb4dae1e1c1d3085567cb6': np.nan,
                           '3ab42b6c81443da2e1f13e0a7d0150416620bf7c': np.nan,
                           '396d3b38be7269b2707f7a4feffae71cba4bb110': 0.0}

        computed_scores = val.compute_challenge_scores(self.challenge)
        for key in expected_scores:
            self.assertTrue(_equals_or_both_nan(expected_scores[key], computed_scores[key]))

    def test_compute_challenge_score(self):
        participant = self.challenge.get_participant("dbcc5a6c8b126e6be68b2bfb838d23ade8e7de57")
        self.assertEqual(1.0, val.compute_challenge_score(participant))

    def test_compute_competition_score(self):
        participant = self.challenge.get_participant("dbcc5a6c8b126e6be68b2bfb838d23ade8e7de57")
        self.assertEqual(0.9, val.compute_competition_score(participant))

    def test_compute_challenge_rewards(self):
        expected_rewards = {'16391fe372c2f5b2a3874e65187dd173e6552fe6': Decimal('0'),
                            'dbcc5a6c8b126e6be68b2bfb838d23ade8e7de57': Decimal('273.9130434782'),
                            '889432c467b59475900845c1f414e0d3461d1459': Decimal('0'),
                            '7e3f69b71019b93123114375119d18df6a1f3a2a': Decimal('0'),
                            '4c2a8ed7374912ee6c980b29fe5021aa94f4c543': Decimal('0'),
                            'b7ece21c962697a3194c0b6e4b81db54433055ab': Decimal('6.0869565217'),
                            'a16a483a18f00a00b14b93438e8ad21f485b382f': Decimal('79.1304347826'),
                            'fe3c3f351e6df846566e10119e4295ed69b2cd72': Decimal('127.8260869565'),
                            'f097ed117dd3930f1eeb91569a6a61b4c52744c8': Decimal('54.7826086956'),
                            '3e04d3e08ea54c34f721ac007107fe7a4bd957d5': Decimal('30.4347826086'),
                            '1be48b2b7594fe7e47701bc7e7bb83beaf84ecc0': Decimal('103.4782608695'),
                            'fb8bc3433016a64a3b8aa23c0275afe7c20feb53': Decimal('225.2173913043'),
                            'eba5ba85f72e40a9f2b8782c2bd51ec31a5564da': Decimal('249.5652173913'),
                            'e14867e41410bd7232a52b9400476c08b4e535b3': Decimal('176.5217391304'),
                            '98e41d65e981a882afcb05d41939197da3465b6f': Decimal('200.8695652173'),
                            '8355b0536a6a3b0116374b33dc3f8b7c344ca8bb': Decimal('0'),
                            '2547002b0b8bfd052111981ef314118b6b54f61c': Decimal('152.1739130434'),
                            'bf72960a4051d7f0a3fb4dae1e1c1d3085567cb6': Decimal('0'),
                            '3ab42b6c81443da2e1f13e0a7d0150416620bf7c': Decimal('0'),
                            '396d3b38be7269b2707f7a4feffae71cba4bb110': Decimal('0')}
        self.assertEqual(expected_rewards, val.compute_challenge_rewards(self.challenge))

    def test_compute_competition_rewards(self):
        expected_rewards = {'16391fe372c2f5b2a3874e65187dd173e6552fe6': Decimal('0'),
                            'dbcc5a6c8b126e6be68b2bfb838d23ade8e7de57': Decimal('1181.25'),
                            '889432c467b59475900845c1f414e0d3461d1459': Decimal('0'),
                            '7e3f69b71019b93123114375119d18df6a1f3a2a': Decimal('0'),
                            '4c2a8ed7374912ee6c980b29fe5021aa94f4c543': Decimal('0'),
                            'b7ece21c962697a3194c0b6e4b81db54433055ab': Decimal('0'),
                            'a16a483a18f00a00b14b93438e8ad21f485b382f': Decimal('0'),
                            'fe3c3f351e6df846566e10119e4295ed69b2cd72': Decimal('236.2499999999'),
                            'f097ed117dd3930f1eeb91569a6a61b4c52744c8': Decimal('0'),
                            '3e04d3e08ea54c34f721ac007107fe7a4bd957d5': Decimal('0'),
                            '1be48b2b7594fe7e47701bc7e7bb83beaf84ecc0': Decimal('78.7499999999'),
                            'fb8bc3433016a64a3b8aa23c0275afe7c20feb53': Decimal('866.25'),
                            'eba5ba85f72e40a9f2b8782c2bd51ec31a5564da': Decimal('1023.75'),
                            'e14867e41410bd7232a52b9400476c08b4e535b3': Decimal('551.2499999999'),
                            '98e41d65e981a882afcb05d41939197da3465b6f': Decimal('708.75'),
                            '8355b0536a6a3b0116374b33dc3f8b7c344ca8bb': Decimal('0'),
                            '2547002b0b8bfd052111981ef314118b6b54f61c': Decimal('393.7499999999'),
                            'bf72960a4051d7f0a3fb4dae1e1c1d3085567cb6': Decimal('0'),
                            '3ab42b6c81443da2e1f13e0a7d0150416620bf7c': Decimal('0'),
                            '396d3b38be7269b2707f7a4feffae71cba4bb110': Decimal('0')}
        self.assertEqual(expected_rewards, val.compute_competition_rewards(self.challenge))

    def test_get_stakes(self):
        expected_stakes = {'16391fe372c2f5b2a3874e65187dd173e6552fe6': Decimal('1000.000000000000000000'),
                           'dbcc5a6c8b126e6be68b2bfb838d23ade8e7de57': Decimal('1000.000000000000000000'),
                           '889432c467b59475900845c1f414e0d3461d1459': Decimal('1000.000000000000000000'),
                           '7e3f69b71019b93123114375119d18df6a1f3a2a': Decimal('1000.000000000000000000'),
                           '4c2a8ed7374912ee6c980b29fe5021aa94f4c543': Decimal('1000.000000000000000000'),
                           'b7ece21c962697a3194c0b6e4b81db54433055ab': Decimal('1000.000000000000000000'),
                           'a16a483a18f00a00b14b93438e8ad21f485b382f': Decimal('1000.000000000000000000'),
                           'fe3c3f351e6df846566e10119e4295ed69b2cd72': Decimal('1000.000000000000000000'),
                           'f097ed117dd3930f1eeb91569a6a61b4c52744c8': Decimal('1000.000000000000000000'),
                           '3e04d3e08ea54c34f721ac007107fe7a4bd957d5': Decimal('1000.000000000000000000'),
                           '1be48b2b7594fe7e47701bc7e7bb83beaf84ecc0': Decimal('1000.000000000000000000'),
                           'fb8bc3433016a64a3b8aa23c0275afe7c20feb53': Decimal('1000.000000000000000000'),
                           'eba5ba85f72e40a9f2b8782c2bd51ec31a5564da': Decimal('1000.000000000000000000'),
                           'e14867e41410bd7232a52b9400476c08b4e535b3': Decimal('1000.000000000000000000'),
                           '98e41d65e981a882afcb05d41939197da3465b6f': Decimal('1000.000000000000000000'),
                           '8355b0536a6a3b0116374b33dc3f8b7c344ca8bb': Decimal('1000.000000000000000000'),
                           '2547002b0b8bfd052111981ef314118b6b54f61c': Decimal('1000.000000000000000000'),
                           'bf72960a4051d7f0a3fb4dae1e1c1d3085567cb6': Decimal('1000.000000000000000000'),
                           '3ab42b6c81443da2e1f13e0a7d0150416620bf7c': Decimal('1000.000000000000000000'),
                           '396d3b38be7269b2707f7a4feffae71cba4bb110': Decimal('1000.000000000000000000')}
        self.assertEqual(expected_stakes, val.get_stakes(self.challenge))

    def test_compute_stake_rewards(self):
        expected_rewards = {'16391fe372c2f5b2a3874e65187dd173e6552fe6': Decimal('84'),
                            'dbcc5a6c8b126e6be68b2bfb838d23ade8e7de57': Decimal('84'),
                            '889432c467b59475900845c1f414e0d3461d1459': Decimal('84'),
                            '7e3f69b71019b93123114375119d18df6a1f3a2a': Decimal('84'),
                            '4c2a8ed7374912ee6c980b29fe5021aa94f4c543': Decimal('84'),
                            'b7ece21c962697a3194c0b6e4b81db54433055ab': Decimal('84'),
                            'a16a483a18f00a00b14b93438e8ad21f485b382f': Decimal('84'),
                            'fe3c3f351e6df846566e10119e4295ed69b2cd72': Decimal('84'),
                            'f097ed117dd3930f1eeb91569a6a61b4c52744c8': Decimal('84'),
                            '3e04d3e08ea54c34f721ac007107fe7a4bd957d5': Decimal('84'),
                            '1be48b2b7594fe7e47701bc7e7bb83beaf84ecc0': Decimal('84'),
                            'fb8bc3433016a64a3b8aa23c0275afe7c20feb53': Decimal('84'),
                            'eba5ba85f72e40a9f2b8782c2bd51ec31a5564da': Decimal('84'),
                            'e14867e41410bd7232a52b9400476c08b4e535b3': Decimal('84'),
                            '98e41d65e981a882afcb05d41939197da3465b6f': Decimal('84'),
                            '8355b0536a6a3b0116374b33dc3f8b7c344ca8bb': Decimal('84'),
                            '2547002b0b8bfd052111981ef314118b6b54f61c': Decimal('84'),
                            'bf72960a4051d7f0a3fb4dae1e1c1d3085567cb6': Decimal('84'),
                            '3ab42b6c81443da2e1f13e0a7d0150416620bf7c': Decimal('84'),
                            '396d3b38be7269b2707f7a4feffae71cba4bb110': Decimal('84')}
        self.assertEqual(expected_rewards, val.compute_stake_rewards(self.challenge))

    def test_get_predictions(self):
        participant = self.challenge.get_participant("dbcc5a6c8b126e6be68b2bfb838d23ade8e7de57")
        predictions = val.get_predictions(participant)
        self.assertEqual('SALT', predictions[0][0])
        self.assertAlmostEqual(0.5262399, predictions[0][1], places=10)
        self.assertEqual('MTX', predictions[-1][0])
        self.assertAlmostEqual(0.5107959, predictions[-1][1], places=10)

    def test_get_requested_assets(self):
        assets = val.get_requested_assets(self.challenge)
        self.assertEqual('SALT', assets[0])
        self.assertEqual('MTX', assets[-1])

    def test_get_values(self):
        values = val.get_values(self.challenge)
        self.assertEqual('AXIS', values[0][0])
        self.assertAlmostEqual(0.1656051, values[0][1], places=7)
        self.assertEqual('MGO', values[-1][0])
        self.assertAlmostEqual(0.06794055, values[-1][1], places=7)


def _equals_or_both_nan(x, y):
    return np.isnan(y) if np.isnan(x) else x == y


if __name__ == "__main__":
    main()
