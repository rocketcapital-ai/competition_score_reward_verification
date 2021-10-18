#!env python

import os
import datetime

from unittest import TestCase, main
from decimal import Decimal
from pathlib import Path
from lib.blockchain import Competition, download_ipfs_file

COMPETITION_NAME = "ROCKET"
COMPETITION_GENESIS_BLOCK = 19664491

DATA_DIR = Path("data")

ALL_TESTS = False  # True for executing slow tests like dataset download or challenge pool read from blockchain


class TestCompetition(TestCase):

    def setUp(self):
        self.competition = Competition(COMPETITION_NAME, COMPETITION_GENESIS_BLOCK, DATA_DIR)

    def test_name(self):
        self.assertEqual(COMPETITION_NAME, self.competition.name)

    def test_genesis_block(self):
        self.assertEqual(COMPETITION_GENESIS_BLOCK, self.competition.genesis_block)

    def test_get_competition_dir(self):
        self.assertEqual(DATA_DIR.joinpath(COMPETITION_NAME), self.competition.get_competition_dir())

    def test_address(self):
        self.assertEqual("0x6cadf1eb6e14650af15194ed3d4c78f585598a70", self.competition.address)

    # this will fail after 11/10/2021 because challenge 1 will not be the latest challenge
    def test_get_latest_challenge_number(self):
        # gets the challenge number based on current time; this may fail for several reasons
        start = datetime.datetime(2021, 10, 4, 6, 0, tzinfo=datetime.timezone.utc)  # challenge starts at 6:00 UTC
        now = datetime.datetime.now(datetime.timezone.utc)
        weeks_since_start = (now - start).days // 7
        self.assertEqual(weeks_since_start + 1, self.competition.get_latest_challenge_number())

    # this will fail after 11/10/2021 because challenge 1 will not be the latest challenge
    #def test_get_current_challenge_pool(self):
    #    self.assertEqual(1680, self.competition.get_current_challenge_pool())

    # this will fail after 11/10/2021 because challenge 1 will not be the latest challenge
    #def test_get_current_competition_pool(self):
    #    self.assertEqual(5040, self.competition.get_current_competition_pool())

    # this will fail after 11/10/2021 because challenge 1 will not be the latest challenge
    #def test_get_current_stake_pool(self):
    #    self.assertEqual(1680, self.competition.get_current_stake_pool())

    def test_get_challenge(self):
        challenge = self.competition.get_challenge(1)
        self.assertEqual(1, challenge.number)


class TestChallenge(TestCase):

    def setUp(self):
        self.challenge = Competition(COMPETITION_NAME, COMPETITION_GENESIS_BLOCK, DATA_DIR).get_challenge(1)

    def test_number(self):
        self.assertEqual(1, self.challenge.number)

    def test_get_competition_dir(self):
        self.assertEqual(DATA_DIR.joinpath(COMPETITION_NAME).joinpath("1"), self.challenge.get_challenge_dir())

    def test_get_dataset_ipfs_cid(self):
        self.assertEqual("QmZteoVtw4bPTLLjQE4x6gHLgHuHBET9d1JX7CtVqWCsKA", self.challenge.get_dataset_ipfs_cid())

    def test_download_dataset_file(self):
        dataset_file = self.challenge.download_dataset_file(force=ALL_TESTS)
        self.assertEqual(251527382, os.stat(dataset_file).st_size)

    def test_get_results_ipfs_cid(self):
        self.assertEqual("QmZ8iHvEWndcxp8nhRCHQKDQAaSeBD1XYA9qobQ28dqiWm", self.challenge.get_results_ipfs_cid())

    def test_download_results_file(self):
        results_file = self.challenge.download_results_file(force=ALL_TESTS)
        self.assertEqual(2194, os.stat(results_file).st_size)

    def test_get_public_key_cid(self):
        self.assertEqual("QmNa67Hjij2s7n3GNuKLFqkjobQJUFHwMKYNwp2xcJFDVs", self.challenge.get_public_key_cid())

    def test_download_public_key(self):
        public_key_file = self.challenge.download_public_key_file(force=ALL_TESTS)
        self.assertEqual(450, os.stat(public_key_file).st_size)

    def test_get_private_key_cid(self):
        self.assertEqual("QmePJ2nJuTrKZw28VGPBoZ8VRcWdKFRVqa8nhLGcroMa8z", self.challenge.get_private_key_cid())

    def test_download_private_key(self):
        private_key_file = self.challenge.download_private_key_file(force=ALL_TESTS)
        self.assertEqual(1703, os.stat(private_key_file).st_size)

    def test_get_submission_counter(self):
        self.assertEqual(17, self.challenge.get_submission_counter())

    def test_get_submitter_addresses(self):
        submitters = ["b7ece21c962697a3194c0b6e4b81db54433055ab", "a16a483a18f00a00b14b93438e8ad21f485b382f",
                      "e14867e41410bd7232a52b9400476c08b4e535b3", "98e41d65e981a882afcb05d41939197da3465b6f",
                      "f097ed117dd3930f1eeb91569a6a61b4c52744c8", "3e04d3e08ea54c34f721ac007107fe7a4bd957d5",
                      "1be48b2b7594fe7e47701bc7e7bb83beaf84ecc0", "fe3c3f351e6df846566e10119e4295ed69b2cd72",
                      "eba5ba85f72e40a9f2b8782c2bd51ec31a5564da"]
        self.assertEqual(submitters, self.challenge.get_submitter_addresses(4, 13))

    def test_get_all_submitter_addresses(self):
        submitters = ["dbcc5a6c8b126e6be68b2bfb838d23ade8e7de57", "889432c467b59475900845c1f414e0d3461d1459",
                      "7e3f69b71019b93123114375119d18df6a1f3a2a", "4c2a8ed7374912ee6c980b29fe5021aa94f4c543",
                      "b7ece21c962697a3194c0b6e4b81db54433055ab", "a16a483a18f00a00b14b93438e8ad21f485b382f",
                      "e14867e41410bd7232a52b9400476c08b4e535b3", "98e41d65e981a882afcb05d41939197da3465b6f",
                      "f097ed117dd3930f1eeb91569a6a61b4c52744c8", "3e04d3e08ea54c34f721ac007107fe7a4bd957d5",
                      "1be48b2b7594fe7e47701bc7e7bb83beaf84ecc0", "fe3c3f351e6df846566e10119e4295ed69b2cd72",
                      "eba5ba85f72e40a9f2b8782c2bd51ec31a5564da", "fb8bc3433016a64a3b8aa23c0275afe7c20feb53",
                      "8355b0536a6a3b0116374b33dc3f8b7c344ca8bb", "2547002b0b8bfd052111981ef314118b6b54f61c",
                      "396d3b38be7269b2707f7a4feffae71cba4bb110"]
        self.assertEqual(submitters, self.challenge.get_all_submitter_addresses())

    def test_get_phase(self):
        self.assertEqual(4, self.challenge.get_phase())

    def test_get_participant(self):
        participant_address = "dbcc5a6c8b126e6be68b2bfb838d23ade8e7de57"
        staker = self.challenge.get_participant(participant_address)
        self.assertEqual(participant_address, staker.address)

    def test_get_stakers(self):
        stakes = {"16391fe372c2f5b2a3874e65187dd173e6552fe6": Decimal("1000.000000000000000000"),
                  "dbcc5a6c8b126e6be68b2bfb838d23ade8e7de57": Decimal("1000.000000000000000000"),
                  "889432c467b59475900845c1f414e0d3461d1459": Decimal("1000.000000000000000000"),
                  "7e3f69b71019b93123114375119d18df6a1f3a2a": Decimal("1000.000000000000000000"),
                  "4c2a8ed7374912ee6c980b29fe5021aa94f4c543": Decimal("1000.000000000000000000"),
                  "b7ece21c962697a3194c0b6e4b81db54433055ab": Decimal("1000.000000000000000000"),
                  "a16a483a18f00a00b14b93438e8ad21f485b382f": Decimal("1000.000000000000000000"),
                  "fe3c3f351e6df846566e10119e4295ed69b2cd72": Decimal("1000.000000000000000000"),
                  "f097ed117dd3930f1eeb91569a6a61b4c52744c8": Decimal("1000.000000000000000000"),
                  "3e04d3e08ea54c34f721ac007107fe7a4bd957d5": Decimal("1000.000000000000000000"),
                  "1be48b2b7594fe7e47701bc7e7bb83beaf84ecc0": Decimal("1000.000000000000000000"),
                  "fb8bc3433016a64a3b8aa23c0275afe7c20feb53": Decimal("1000.000000000000000000"),
                  "eba5ba85f72e40a9f2b8782c2bd51ec31a5564da": Decimal("1000.000000000000000000"),
                  "e14867e41410bd7232a52b9400476c08b4e535b3": Decimal("1000.000000000000000000"),
                  "98e41d65e981a882afcb05d41939197da3465b6f": Decimal("1000.000000000000000000"),
                  "8355b0536a6a3b0116374b33dc3f8b7c344ca8bb": Decimal("1000.000000000000000000"),
                  "2547002b0b8bfd052111981ef314118b6b54f61c": Decimal("1000.000000000000000000"),
                  "bf72960a4051d7f0a3fb4dae1e1c1d3085567cb6": Decimal("1000.000000000000000000"),
                  "3ab42b6c81443da2e1f13e0a7d0150416620bf7c": Decimal("1000.000000000000000000"),
                  "396d3b38be7269b2707f7a4feffae71cba4bb110": Decimal("1000.000000000000000000")}
        self.assertEqual(set(stakes), set(self.challenge.get_stakes()))

    def test_get_all_stakers(self):
        challenge_staker_addresses = {address for address in self.challenge.get_stakes()}
        all_staker_addresses = {staker.address for staker in self.challenge.get_all_participants()}
        self.assertEqual(challenge_staker_addresses, all_staker_addresses)

    def test_get_stake_pool(self):
        self.assertEqual(Decimal(1680), self.challenge.get_stake_pool())

    def test_get_challenge_pool(self):
        self.assertEqual(Decimal(1680), self.challenge.get_challenge_pool())

    def test_get_competition_pool(self):
        self.assertEqual(Decimal(5040), self.challenge.get_competition_pool())


class TestParticipant(TestCase):

    def setUp(self):
        self.challenge = Competition(COMPETITION_NAME, COMPETITION_GENESIS_BLOCK, DATA_DIR).get_challenge(1)
        self.participant = self.challenge.get_participant("dbcc5a6c8b126e6be68b2bfb838d23ade8e7de57")

    def test_address(self):
        self.assertEqual("dbcc5a6c8b126e6be68b2bfb838d23ade8e7de57", self.participant.address)

    def test_get_stake(self):
        self.assertEqual(Decimal(1000), self.participant.get_stake())

    def test_get_submitter_stake(self):
        self.assertEqual(Decimal(1000), self.participant.get_submitter_stake())

    def test_get_staking_reward(self):
        self.assertEqual(Decimal("84"), self.participant.get_staking_reward())

    def test_get_submission_ipfs_cid(self):
        self.assertEqual("Qme4EwAWYSkamc8LUp6qBSt18bQ4UnMHLMA4cazmhRQLnc", self.participant.get_submission_ipfs_cid())

    def test_download_submission_file(self):
        submission_file = self.participant.download_submission_file(force=ALL_TESTS)
        self.assertEqual(8583, os.stat(submission_file).st_size)

    def test_get_challenge_reward(self):
        self.assertEqual(Decimal("273.913043478199995206"), self.participant.get_challenge_reward())

    def test_get_competition_reward(self):
        self.assertEqual(Decimal("1181.250000000000000000"), self.participant.get_competition_reward())

    def test_get_challenge_score(self):
        self.assertEqual(Decimal("1"), self.participant.get_challenge_score())

    def test_get_competition_score(self):
        self.assertEqual(Decimal("0.900000000000000022"), self.participant.get_competition_score())


class Test(TestCase):

    def setUp(self):
        self.challenge = Competition(COMPETITION_NAME, COMPETITION_GENESIS_BLOCK, DATA_DIR).get_challenge(1)

    def test_download_ipfs_file(self):
        if ALL_TESTS:
            dataset_ipfs_cid = self.challenge.get_dataset_ipfs_cid()
            challenge_dir = self.challenge.get_challenge_dir()
            dataset_file = challenge_dir.joinpath(dataset_ipfs_cid + ".zip")
            downloaded_file = download_ipfs_file(dataset_ipfs_cid, challenge_dir, suffix=".zip")
            self.assertEqual(dataset_file, downloaded_file)
            self.assertEqual(251527382, os.stat(downloaded_file).st_size)


if __name__ == "__main__":
    main()
