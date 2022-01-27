from __future__ import annotations

import numpy as np
import math
import scipy.stats.mstats as mstats

from abc import ABC, abstractmethod
from decimal import Decimal, ROUND_DOWN
from sklearn.metrics import mean_squared_error

"""
module to compute scores and rewards according to challenge number

NOTE: exact values are represented as Decimals
      approximate values are represented as floats
"""


def validate_prediction(assets: [str], predictions: [(str, Decimal)]) -> bool:
    """checks if a prediction from a participant is valid

    :param assets: [str]
        the list of assets whose values must be predicted
    :param predictions: [(str, Decimal)]
        the list of (asset, value) pairs sent by the participant
    :return: bool
        true iff predictions contain all assets once and only once, and prediction exists for assets not in the list
    """
    predicted_assets = set([asset for asset, value in predictions])

    # check if all assets have been predicted and prediction exists for assets not in the list
    if set(assets) != predicted_assets:
        return False

    # check if all assets have been predicted only once
    if len(assets) != len(predictions):
        return False

    # this can only happen when assets are repeated in the dataset (should happen!)
    if len(set(assets)) != len(assets):
        return False

    # check if there is any NaN in the predicted values
    return not any((math.isnan(value) for asset, value in predictions))


def compute_challenge_error(challenge_number: int, predictions: [Decimal], assets_values: [Decimal]) -> float:
    """computes the challenge error of a participant

    :param challenge_number: int
        the challenge number
    :param predictions: [Decimal]
        the list of predictions, ordered by assets
    :param assets_values:
        the list of correct values, ordered by assets
    :return: float
        the Root Mean Square Error between predictions and values
    """
    scorer = Scorer.get(challenge_number)
    return scorer.compute_challenge_error(predictions, assets_values)


def compute_challenge_scores(challenge_number: int, participants_errors: [float]) -> [float]:
    """computes the challenge scores of all participants to a challenge

    :param challenge_number: int
        the challenge number
    :param participants_errors: [float]
        the list of errors of all participants
    :return: [float]
        the list of challenge scores of all participants
    """
    scorer = Scorer.get(challenge_number)
    return scorer.compute_challenge_scores(participants_errors)


def compute_competition_score(challenge_number: int, challenge_scores: [float]) -> float:
    """computes the competition score of a participant

    :param challenge_number: int
        the challenge number
    :param challenge_scores: [float]
        the list challenge scores of the participant, from challenge 1
    :return: float
        the participant competition score
    """
    scorer = Scorer.get(challenge_number)
    return scorer.compute_competition_score(challenge_scores)


def compute_challenge_rewards(challenge_number: int, challenge_scores: [float], challenge_pool: Decimal) -> [Decimal]:
    """ computes the challenge rewards of all participants

    :param challenge_number: int
        the challenge number
    :param challenge_scores: [float]
        the challenge scores of all participants
    :param challenge_pool: Decimal
        the total sum to be paid for challenge rewards
    :return: [Decimal]
        the challenge rewards of all participants
    """
    scorer = Scorer.get(challenge_number)
    return scorer.compute_challenge_rewards(challenge_scores, challenge_pool)


def compute_competition_rewards(challenge_number: int, competition_scores: [float],
                                competition_pool: Decimal) -> [Decimal]:
    """ computes the competition rewards of all participants

    :param challenge_number: int
        the challenge number
    :param competition_scores: [float]
        the competition scores of all participants
    :param competition_pool: Decimal
        the total sum to be paid for competition rewards
    :return: [Decimal]
        the competition rewards of all participants
    """
    scorer = Scorer.get(challenge_number)
    return scorer.compute_competition_rewards(competition_scores, competition_pool)


def compute_stake_rewards(challenge_number: int, stakes: [Decimal], stake_pool: Decimal) -> [Decimal]:
    """ computes the stake rewards of all participants

    :param challenge_number: int
        the challenge number
    :param stakes: [Decimal]
        the stakes of all participants
    :param stake_pool: Decimal
        the total sum to be paid for stake rewards
    :return: [Decimal]
        the stake rewards of all participants
    """
    scorer = Scorer.get(challenge_number)
    return scorer.compute_stake_rewards(stakes, stake_pool)


def compute_challenge_pool(challenge_number: int, num_predictors: int) -> Decimal:
    """computes the pool to pay challenge rewards for a given challenge

    :param challenge_number: int
        the challenge number
    :param num_predictors: int
        the total number of participants sending predictions at the previous challenge
    :return: Decimal
        the challenge pool
    """
    scorer = Scorer.get(challenge_number)
    return scorer.compute_challenge_pool(num_predictors)


def compute_competition_pool(challenge_number: int, num_predictors: int) -> Decimal:
    """computes the pool to pay competition rewards for a given challenge

    :param challenge_number: int
        the challenge number
    :param num_predictors: int
        the total number of participants sending predictions at the previous challenge
    :return: Decimal
        the challenge pool
    """
    scorer = Scorer.get(challenge_number)
    return scorer.compute_competition_pool(num_predictors)


def compute_stake_pool(challenge_number: int, num_predictors: int, num_stakers: int) -> Decimal:
    """ computes the pool to pay stake rewards for a given challenge

    :param challenge_number: int
        the challenge number
    :param num_predictors: int
        the total number of participants sending predictions at the previous challenge
    :param num_stakers: int
        the total number of stakers at the previous challenge
    :return: Decimal
        the stake pool
    """
    scorer = Scorer.get(challenge_number)
    return scorer.compute_stake_pool(num_predictors, num_stakers)


def compute_pool_surplus(challenge_number: int, num_predictors: int, num_stakers: int) -> Decimal:
    """computes the remaining surplus of the total pool

    :param challenge_number: int
        the challenge number
    :param num_predictors: int
        the total number of participants sending predictions at the previous challenge
    :param num_stakers: int
        the total number of stakers at the previous challenge
    :return: Decimal
        the remaining surplus
    """
    scorer = Scorer.get(challenge_number)
    return scorer.compute_pool_surplus(num_predictors, num_stakers)


def dec(x):
    return Decimal(x)


class Scorer (ABC):
    """
    abstract class to compute scores and rewards
    """

    TOTAL_WEEKLY_POOL = dec(200000)
    REWARD_PRECISION = "0.0000000001"  # 10 decimal digits

    @abstractmethod
    def compute_challenge_error(self, predictions: [Decimal], assets_values: [Decimal]) -> float:
        pass

    @abstractmethod
    def compute_challenge_scores(self, participants_errors: [float]) -> [float]:
        pass

    @abstractmethod
    def compute_competition_score(self, challenge_scores: [float]) -> float:
        pass

    @abstractmethod
    def compute_challenge_rewards(self, challenge_scores: [float], challenge_pool: Decimal) -> [Decimal]:
        pass

    @abstractmethod
    def compute_competition_rewards(self, competition_scores: [float], competition_pool: Decimal) -> [Decimal]:
        pass

    @abstractmethod
    def compute_stake_rewards(self, stakes: [Decimal], stake_pool: Decimal) -> [Decimal]:
        pass

    @abstractmethod
    def compute_challenge_pool(self, num_predictors: int) -> Decimal:
        pass

    @abstractmethod
    def compute_competition_pool(self, num_predictors: int) -> Decimal:
        pass

    @abstractmethod
    def compute_stake_pool(self, num_predictors: int, num_stakers: int) -> Decimal:
        pass

    def compute_pool_surplus(self, num_predictors: int, num_stakers: int) -> Decimal:
        return Scorer.TOTAL_WEEKLY_POOL - (self.compute_challenge_pool(num_predictors) +
                                           self.compute_competition_pool(num_predictors) +
                                           self.compute_stake_pool(num_predictors, num_stakers))

    @abstractmethod
    def get_std_dev_penalty(self):
        pass

    @abstractmethod
    def get_skip_penalty(self):
        pass

    @abstractmethod
    def get_window_size(self):
        pass

    @staticmethod
    def get(challenge_number: int) -> Scorer:
        """returns the scorer valid at a given challenge

        :param challenge_number:
            the challenge number
        :return:

        """
        if challenge_number < 0:
            raise LookupError()
        elif challenge_number <= 4:
            return ScorerFrom1To4()
        elif challenge_number == 5:
            return ScorerAt5()
        elif challenge_number <= 17:
            return ScorerFrom6To17()
        else:
            return ScorerFrom18()


class Scorer1 (Scorer, ABC):
    """
    first implementation of Scorer compliant with
    https://app.gitbook.com/@rocket-capital-investment/s/rci-competition/scoring-and-reward-policy
    """

    UNIT_WEEKLY_POOL = dec(200)  # reach total at 1000 submitters/stakers
    CHALLENGE_REWARD_PERC = dec("0.2")
    COMPETITION_REWARD_PERC = dec("0.6")
    STAKE_REWARD_PERC = dec("0.2")

    def compute_challenge_error(self, predictions: [Decimal], assets_values: [Decimal]) -> float:
        float_predictions = np.array(predictions, dtype=float)
        float_assets_values = np.array(assets_values, dtype=float)
        return mean_squared_error(float_predictions, float_assets_values, squared=False)

    def compute_challenge_scores(self, participants_errors: [float]) -> [float]:
        n = np.count_nonzero(~np.isnan(participants_errors))
        if n == 0:
            return [np.nan] * len(participants_errors)

        ranks = mstats.rankdata(np.ma.masked_invalid(participants_errors))
        ranks[ranks == 0] = np.nan
        return [(n - r) / (n - 1) for r in ranks]

    def compute_competition_score(self, challenge_scores: [float]) -> float:
        window_size = self.get_window_size()
        scores = challenge_scores[-window_size:]
        num_skips = window_size - len(scores)
        for score in scores:
            if math.isnan(score):
                num_skips = num_skips + 1
        if num_skips == window_size:
            return np.nan
        a = np.nanmean(scores)
        b = self.get_std_dev_penalty() * 2 * np.nanstd(scores)
        c = self.get_skip_penalty() * num_skips / (window_size - 1)
        return max(a - (b + c), 0)

    def compute_challenge_rewards(self, challenge_scores: [float], challenge_pool: Decimal) -> [Decimal]:
        challenge_scores = [score if not np.isnan(score) else 0 for score in challenge_scores]
        factors = [max(dec(score) - dec("0.25"), dec(0)) for score in challenge_scores]
        return Scorer1.__distribute(factors, challenge_pool)

    def compute_competition_rewards(self, competition_scores: [float], competition_pool: Decimal) -> [Decimal]:
        n = np.count_nonzero(~np.isnan(competition_scores))
        if n == 0:
            return [0] * len(competition_scores)

        ranks = mstats.rankdata(np.ma.masked_invalid(competition_scores))
        ranks[ranks == 0] = np.nan
        ranks = [(r - 1) / (n - 1) if not np.isnan(r) else 0 for r in ranks]
        factors = [max(dec(rank) - dec("0.5"), dec(0)) for rank in ranks]
        return Scorer1.__distribute(factors, competition_pool)

    def compute_stake_rewards(self, stakes: [Decimal], stake_pool: Decimal) -> [Decimal]:
        return Scorer1.__distribute(stakes, stake_pool)

    def compute_challenge_pool(self, num_predictors: int) -> Decimal:
        max_pool = Scorer.TOTAL_WEEKLY_POOL * Scorer1.CHALLENGE_REWARD_PERC
        pool = Scorer1.UNIT_WEEKLY_POOL * Scorer1.CHALLENGE_REWARD_PERC * num_predictors
        return min(pool, max_pool)

    def compute_competition_pool(self, num_predictors: int) -> Decimal:
        max_pool = Scorer.TOTAL_WEEKLY_POOL * Scorer1.COMPETITION_REWARD_PERC
        pool = Scorer1.UNIT_WEEKLY_POOL * Scorer1.COMPETITION_REWARD_PERC * num_predictors
        return min(pool, max_pool)

    def compute_stake_pool(self, num_predictors: int, num_stakers: int) -> Decimal:
        max_pool = Scorer.TOTAL_WEEKLY_POOL * Scorer1.STAKE_REWARD_PERC
        pool = Scorer1.UNIT_WEEKLY_POOL * Scorer1.STAKE_REWARD_PERC * num_predictors
        return min(pool, max_pool)

    @staticmethod
    def __distribute(factors: [Decimal], pool: Decimal) -> [Decimal]:
        total = sum(factors)
        return [((pool * factor) / total).quantize(Decimal(Scorer.REWARD_PRECISION), rounding=ROUND_DOWN).normalize()
                for factor in factors]


class ScorerFrom1To4 (Scorer1):
    """valid from challenge 1 to challenge 4"""

    STDDEV_PENALTY = 0.1
    SKIP_PENALTY = 0.1
    WINDOW_SIZE = 4

    def compute_stake_pool(self, num_predictors: int, num_stakers: int) -> Decimal:
        max_pool = Scorer.TOTAL_WEEKLY_POOL * Scorer1.STAKE_REWARD_PERC
        pool = Scorer1.UNIT_WEEKLY_POOL * Scorer1.STAKE_REWARD_PERC * num_stakers
        return min(pool, max_pool)

    def get_std_dev_penalty(self):
        return self.STDDEV_PENALTY

    def get_skip_penalty(self):
        return self.SKIP_PENALTY

    def get_window_size(self):
        return self.WINDOW_SIZE


# used only for challenge 5, where a bug in the backoffice software caused all submission to be invalid
class ScorerAt5 (Scorer1):
    """valid just for challenge 5"""

    # 28 submissions for challenge 5
    CHALLENGE_5_PREDICTORS = 28

    # challenge rewards are the same for all; an extra has been directly sent to participant wallets
    # to give all the same they would have got if they were arrived first
    def compute_challenge_rewards(self, challenge_scores: [float], challenge_pool: Decimal) -> [Decimal]:
        return [(challenge_pool / ScorerAt5.CHALLENGE_5_PREDICTORS)
                .quantize(Decimal(Scorer.REWARD_PRECISION), rounding=ROUND_DOWN).normalize()] \
               * ScorerAt5.CHALLENGE_5_PREDICTORS

    # competition rewards are the same for all; an extra has been directly sent to participant wallets
    # to give all the same they would have got if they were arrived first
    def compute_competition_rewards(self, competition_scores: [float], competition_pool: Decimal) -> [Decimal]:
        return [(competition_pool / ScorerAt5.CHALLENGE_5_PREDICTORS)
                .quantize(Decimal(Scorer.REWARD_PRECISION), rounding=ROUND_DOWN).normalize()]\
               * ScorerAt5.CHALLENGE_5_PREDICTORS

    # override num_predictors, which would be zero because all submissions are invalid
    def compute_challenge_pool(self, num_predictors: int) -> Decimal:
        return dec(Scorer1.UNIT_WEEKLY_POOL * Scorer1.CHALLENGE_REWARD_PERC * ScorerAt5.CHALLENGE_5_PREDICTORS)

    # override num_predictors, which would be zero because all submissions are invalid
    def compute_competition_pool(self, num_predictors: int) -> Decimal:
        return dec(Scorer1.UNIT_WEEKLY_POOL * Scorer1.COMPETITION_REWARD_PERC * ScorerAt5.CHALLENGE_5_PREDICTORS)

    # override num_predictors, which would be zero because all submissions are invalid
    def compute_stake_pool(self, num_predictors: int, num_stakers: int) -> Decimal:
        return dec(Scorer1.UNIT_WEEKLY_POOL * Scorer1.STAKE_REWARD_PERC * ScorerAt5.CHALLENGE_5_PREDICTORS)

    def get_std_dev_penalty(self):
        pass

    def get_skip_penalty(self):
        pass

    def get_window_size(self):
        pass


class ScorerFrom6To17 (Scorer1):
    """valid from challenge 6 to challenge X"""

    STDDEV_PENALTY = 0.1
    SKIP_PENALTY = 0.1
    WINDOW_SIZE = 4

    def get_std_dev_penalty(self):
        return self.STDDEV_PENALTY

    def get_skip_penalty(self):
        return self.SKIP_PENALTY

    def get_window_size(self):
        return self.WINDOW_SIZE


class ScorerFrom18 (Scorer1):
    """valid from challenge 18 on"""

    STDDEV_PENALTY = 0.2
    SKIP_PENALTY = 0.5
    WINDOW_SIZE = 8

    def get_std_dev_penalty(self):
        return self.STDDEV_PENALTY

    def get_skip_penalty(self):
        return self.SKIP_PENALTY

    def get_window_size(self):
        return self.WINDOW_SIZE
