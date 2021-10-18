
import requests
import base58
import ipfshttpclient
import multiaddr
import decimal
import csv

from _pysha3 import keccak_256
from decimal import Decimal
from typing import Any
from collections import defaultdict
from pathlib import Path

REGISTRY_ADDRESS = "0x0Ee5AFF42564C0D293164b39D85653666ae151Eb"
IPFS_GATEWAY = multiaddr.Multiaddr("/dns/gateway.pinata.cloud/tcp/443/https")
POLYGON_GATEWAY = "https://rpc-mainnet.matic.quiknode.pro"
MAX_RETRIES = 20
REST_PERIOD = 2

NULL_IPFS_CID = "QmNLei78zWmzUdbeRB3CiUfAizWUrbeeZh5K1rhAQKCh51"

"""classes and methods to interact with solidity contract

see https://github.com/rocketcapital-ai/competition/blob/main/contracts/Competition.sol
"""


class Competition:

    def __init__(self, name: str, genesis_block: int, data_dir: Path):
        self._name = name
        self._genesis_block = genesis_block
        self._data_dir = data_dir
        self._address = get_competition_address(name)

    @property
    def name(self):
        return self._name

    @property
    def genesis_block(self):
        return self._genesis_block

    @property
    def address(self):
        return self._address

    def get_competition_dir(self):
        """returns the competition directory, a directory named as the competition inside the data directory"""

        return self._data_dir.joinpath(self.name)

    def get_latest_challenge_number(self) -> int:
        """calls solidity method getLatestChallengeNumber()"""

        challenge_number_uint = self.call("getLatestChallengeNumber()")
        return hex_to_int(challenge_number_uint)

    def get_current_challenge_pool(self) -> Decimal:
        """calls solidity method getCurrentChallengeRewardsBudget()"""

        current_challenge_pool_uint = self.call("getCurrentChallengeRewardsBudget()")
        return hex_to_decimal(current_challenge_pool_uint)

    def get_current_competition_pool(self) -> Decimal:
        """calls solidity method getCurrentTournamentRewardsBudget()"""

        current_competition_pool_uint = self.call("getCurrentTournamentRewardsBudget()")
        return hex_to_decimal(current_competition_pool_uint)

    def get_current_stake_pool(self) -> Decimal:
        """calls solidity method getCurrentStakingRewardsBudget()"""

        current_stake_pool_uint = self.call("getCurrentStakingRewardsBudget()")
        return hex_to_decimal(current_stake_pool_uint)

    def get_challenge(self, challenge_number: int) -> "Challenge":
        return Challenge(challenge_number, self)

    def call(self, fn_signature: str, *args):
        return call(self._address, fn_signature, *args)


class Challenge:

    def __init__(self, number: int, competition: Competition):
        self._number = number
        self._competition = competition

        # these data are loaded from blockchain or from local file
        self._stakes = None
        self._stake_pool = None
        self._challenge_pool = None
        self._competition_pool = None

    @property
    def number(self):
        return self._number

    @property
    def competition(self):
        return self._competition

    def get_challenge_dir(self):
        """returns the challenge directory, named as the challenge number inside the competition directory"""

        return self._competition.get_competition_dir().joinpath(str(self._number))

    def get_dataset_ipfs_cid(self) -> str:
        """calls solidity method getDatasetHash() and converts hash to cid"""

        dataset_hash = self.call("getDatasetHash(uint32)", int_to_uint(self._number))
        return hash_to_cid(dataset_hash)

    def download_dataset_file(self, force=False, verbose=True) -> Path:
        """utility method to read the dataset cid and download the dataset file in the challenge directory"""

        dataset_ipfs_cid = self.get_dataset_ipfs_cid()
        challenge_dir = self.get_challenge_dir()
        return download_ipfs_file(dataset_ipfs_cid, challenge_dir, prefix="dataset-", suffix=".zip", force=force, verbose=verbose)

    def get_results_ipfs_cid(self) -> str:
        """calls solidity method getResultsHash() and converts hash to cid"""

        results_hash = self.call("getResultsHash(uint32)", int_to_uint(self._number))
        return hash_to_cid(results_hash)

    def download_results_file(self, force=False, verbose=True) -> Path:
        """utility method to read the results cid and download the results file in the challenge directory"""

        results_ipfs_cid = self.get_results_ipfs_cid()
        challenge_dir = self.get_challenge_dir()
        return download_ipfs_file(results_ipfs_cid, challenge_dir, prefix="results-", suffix=".csv", force=force, verbose=verbose)

    def get_public_key_cid(self) -> str:
        """calls solidity method getKeyHash()"""

        public_key_hash = self.call("getKeyHash(uint32)", int_to_uint(self._number))
        return hash_to_cid(public_key_hash)

    def download_public_key_file(self, force=False, verbose=True) -> Path:
        """utility method to read the public key cid and download the public key file in the challenge directory"""

        public_key_cid = self.get_public_key_cid()
        challenge_dir = self.get_challenge_dir()
        return download_ipfs_file(public_key_cid, challenge_dir, prefix="public-key-", suffix=".pem", force=force, verbose=verbose)

    def get_private_key_cid(self) -> str:
        """calls solidity method getPrivateKeyHash()"""

        private_key_hash = self.call("getPrivateKeyHash(uint32)", int_to_uint(self._number))
        return hash_to_cid(private_key_hash)

    def download_private_key_file(self, force=False, verbose=True) -> Path:
        """utility method to read the private key cid and download the private key file in the challenge directory"""

        private_key_cid = self.get_private_key_cid()
        challenge_dir = self.get_challenge_dir()
        return download_ipfs_file(private_key_cid, challenge_dir, prefix="private-key-", suffix=".pem", force=force, verbose=verbose)

    def get_submission_counter(self) -> int:
        """calls solidity method getSubmissionCounter()"""

        return hex_to_int(self.call("getSubmissionCounter(uint32)", int_to_uint(self._number)))

    def get_submitter_addresses(self, start_index: int, end_index: int) -> [str]:
        """calls solidity method getSubmitters()"""

        result = self.call("getSubmitters(uint32,uint256,uint256)", int_to_uint(self._number),
                           int_to_uint(start_index), int_to_uint(end_index))
        # skip first 130 chars (0x + start index (32 bytes) and # of items (32 bytes))
        result = result[130:]
        return [result[i+24:i+64] for i in range(0, len(result), 64)]

    def get_all_submitter_addresses(self) -> [str]:
        """utility method to get all submitters"""
        num_submitters = self.get_submission_counter()
        return self.get_submitter_addresses(0, num_submitters)

    def get_phase(self) -> int:
        """calls solidity method getPhase()"""

        return hex_to_int(self.call("getPhase(uint32)",
                                    int_to_uint(self._number)))

    def get_stakes(self) -> {str: Decimal}:
        """returns all the <address, stake> pairs of the challenge"""

        if self._stakes is None:
            self._load_blockchain_info()
        return self._stakes

    def get_participant(self, address: str) -> "Participant":
        return Participant(address, self)

    def get_all_participants(self) -> ["Participant"]:
        """utility method to return all participants"""

        return [self.get_participant(address) for address in self.get_stakes()]

    def get_stake_pool(self) -> Decimal:
        """returns the stake pool"""

        if self._stake_pool is None:
            self._load_blockchain_info()
        return self._stake_pool

    def get_challenge_pool(self) -> Decimal:
        """returns the challenge pool"""

        if self._challenge_pool is None:
            self._load_blockchain_info()
        return self._challenge_pool

    def get_competition_pool(self) -> Decimal:
        """returns the competition pool"""

        if self._competition_pool is None:
            self._load_blockchain_info()
        return self._competition_pool

    def call(self, fn_signature: str, *args):
        return self._competition.call(fn_signature, *args)

    def _load_blockchain_info(self) -> None:
        """scans the blockchain if needed and initializes the variables"""

        stakes_info_file = self.get_challenge_dir().joinpath("_stakes.csv")
        pools_info_file = self.get_challenge_dir().joinpath("_pools.csv")
        if not stakes_info_file.exists() or not pools_info_file.exists():
            self.get_challenge_dir().mkdir(parents=True, exist_ok=True)

            # load blockchain info from blockchain and write to file
            self._scan_blockchain()
            with open(stakes_info_file, "w") as fout:
                writer = csv.writer(fout)
                writer.writerow(["address", "stake"])
                for address, value in self._stakes.items():
                    writer.writerow([address, str(value)])
            with open(pools_info_file, "w") as fout:
                writer = csv.writer(fout)
                writer.writerow(["pool", "amount"])
                writer.writerow(["stake_pool", str(self._stake_pool)])
                writer.writerow(["challenge_pool", str(self._challenge_pool)]),
                writer.writerow(["competition_pool", str(self._competition_pool)])
        else:
            # load blockchain info from file
            with open(stakes_info_file) as fin:
                reader = csv.reader(fin)
                next(reader)
                self._stakes = {address: Decimal(stake) for address, stake in reader}
            with open(pools_info_file) as fin:
                reader = csv.reader(fin)
                next(reader)
                csv_dict = {name: value for name, value in reader}
                self._stake_pool = Decimal(csv_dict.get("stake_pool"))
                self._challenge_pool = Decimal(csv_dict.get("challenge_pool"))
                self._competition_pool = Decimal(csv_dict.get("competition_pool"))

    def _scan_blockchain(self) -> None:
        latest_challenge = self._competition.get_latest_challenge_number()
        assert self._number <= latest_challenge, f"challenge {self._number} has not been opened"

        chunk = 1000
        quantize = Decimal("1e-18")
        latest_block = get_latest_block()
        current_challenge = None

        # track balance of rewards pool
        total_rewards_pool_uint = 0
        current_challenge_percentage = Decimal("0.2")
        current_competition_percentage = Decimal("0.6")

        # dictionaries to track and record values by challenge
        stake_pool = {}
        challenge_pool = {}
        competition_pool = {}
        challenge_percentage = {1: current_challenge_percentage}
        competition_percentage = {1: current_competition_percentage}
        stakes = defaultdict(lambda: Decimal("0"))
        challenge = {}

        # event signatures to look out for
        sponsor_sig = str_to_fn_id("Sponsor(address,uint256,uint256)", True)
        remainder_moved_sig = str_to_fn_id("RemainderMovedToPool(uint256)", True)
        total_rewards_paid_sig = str_to_fn_id("TotalRewardsPaid(uint32,uint256,uint256,uint256)", True)
        challenge_opened_sig = str_to_fn_id("ChallengeOpened(uint32)", True)
        submission_closed_sig = str_to_fn_id("SubmissionClosed(uint32)", True)
        challenge_percentage_sig = str_to_fn_id("ChallengeRewardsPercentageInWeiUpdated(uint256)", True)
        competition_percentage_sig = str_to_fn_id("TournamentRewardsPercentageInWeiUpdated(uint256)", True)
        stake_increased_sig = str_to_fn_id("StakeIncreased(address,uint256)", True)
        stake_decreased_sig = str_to_fn_id("StakeDecreased(address,uint256)", True)
        rewards_payment_sig = str_to_fn_id("RewardsPayment(uint32,address,uint256,uint256,uint256)", True)

        print("scanning the Polygon blockchain")
        for i in range(self._competition.genesis_block, latest_block, chunk):
            print(".", end="", flush=True)

            transactions = scan(i, i + chunk - 1, self._competition.address)
            for tx in transactions:
                topics = tx["topics"]
                sig = topics[0]
                if sig == challenge_percentage_sig:
                    current_challenge_percentage = hex_to_decimal(topics[1])
                elif sig == competition_percentage_sig:
                    current_competition_percentage = hex_to_decimal(topics[1])
                elif sig == sponsor_sig:
                    sponsored_amount = hex_to_int(topics[2])
                    pool_total = hex_to_int(topics[3])
                    total_rewards_pool_uint += sponsored_amount
                    assert total_rewards_pool_uint == pool_total, \
                        f"\nmisalignment in pool total; expected {total_rewards_pool_uint} but got {pool_total}"
                elif sig == remainder_moved_sig:
                    remainder_added_to_pool = hex_to_int(topics[1])
                    total_rewards_pool_uint += remainder_added_to_pool
                elif sig == total_rewards_paid_sig:
                    staking_rewards_paid = hex_to_int(topics[1])
                    challenge_rewards_paid = hex_to_int(topics[2])
                    competition_rewards_paid = hex_to_int(topics[3])
                    total_rewards_pool_uint -= staking_rewards_paid + challenge_rewards_paid + competition_rewards_paid
                elif sig == challenge_opened_sig:
                    # when a challenge is opened, this is the point where the
                    # staking, challenge and competition pools are set
                    current_challenge = hex_to_int(topics[1])
                    this_challenge_pool = Decimal(total_rewards_pool_uint) * current_challenge_percentage * quantize
                    challenge_pool[current_challenge] = this_challenge_pool.quantize(quantize, rounding=decimal.ROUND_DOWN)
                    this_competition_pool = Decimal(total_rewards_pool_uint) * current_competition_percentage * quantize
                    competition_pool[current_challenge] = this_competition_pool.quantize(quantize, rounding=decimal.ROUND_DOWN)
                    this_stake_pool = (Decimal(total_rewards_pool_uint) * quantize) - challenge_pool[current_challenge] - competition_pool[current_challenge]
                    stake_pool[current_challenge] = this_stake_pool.quantize(quantize, rounding=decimal.ROUND_DOWN)
                    challenge_percentage[current_challenge] = current_challenge_percentage
                    competition_percentage[current_challenge] = current_competition_percentage
                elif sig == stake_increased_sig:
                    address = topics[1][-40:]
                    stakes[address] += hex_to_decimal(topics[2])
                elif sig == stake_decreased_sig:
                    address = topics[1][-40:]
                    stakes[address] -= hex_to_decimal(topics[2])
                elif sig == rewards_payment_sig:
                    address = topics[1][-40:]
                    stake_reward = tx["data"][-64:]
                    stakes[address] += hex_to_decimal(topics[2]) + hex_to_decimal(topics[3]) + hex_to_decimal(stake_reward)
                elif sig == submission_closed_sig:
                    closed_submission = hex_to_int(topics[1])
                    assert current_challenge == closed_submission, \
                        f"\nmisalignment with challenge numbers; expected {current_challenge} got {closed_submission}"
                    challenge[current_challenge] = stakes.copy()
                    # terminate if we have the information for the specified challenge
                    if current_challenge == self._number:
                        print("\nblockchain scan completed")
                        # sanity check if specified challenge is the latest challenge
                        if latest_challenge == self._number:
                            current_stake_pool = self._competition.get_current_stake_pool()
                            current_challenge_pool = self._competition.get_current_challenge_pool()
                            current_competition_pool = self._competition.get_current_competition_pool()
                            assert stake_pool[self._number] == current_stake_pool, \
                                f"stake pool misalignment; expected {stake_pool[self._number]} got {current_stake_pool}"
                            assert challenge_pool[self._number] == current_challenge_pool, \
                                f"challenge pool misalignment; expected {challenge_pool[self._number]} got {current_challenge_pool}"
                            assert competition_pool[self._number] == current_competition_pool, \
                                f"competition pool misalignment; expected {competition_pool[self._number]} got {current_competition_pool}"
                        self._stakes = challenge[self._number]
                        self._stake_pool = stake_pool[self._number]
                        self._challenge_pool = challenge_pool[self._number]
                        self._competition_pool = competition_pool[self._number]
                        return

        assert False, f"could not determine pools for challenge {self._number}"


class Participant:

    def __init__(self, address: str, challenge: Challenge):
        self._address = address
        self._challenge = challenge

    @property
    def address(self):
        return self._address

    @property
    def challenge(self):
        return self._challenge

    def get_submitter_stake(self) -> Decimal:
        """calls solidity method getStakedAmountForChallenge()

        NOTE: it returns 0 for if the staker did not submit a prediction for the challenge
        """

        result = self._call("getStakedAmountForChallenge(uint32,address)",
                            int_to_uint(self._challenge.number),
                            address_to_uint(self._address))
        return hex_to_decimal(result)

    def get_stake(self) -> Decimal:
        """returns the stake read from the blockchain"""
        return self._challenge.get_stakes().get(self._address)

    def get_staking_reward(self) -> Decimal:
        """calls solidity method getStakingRewards()"""

        result = self._call("getStakingRewards(uint32,address)",
                            int_to_uint(self._challenge.number),
                            address_to_uint(self._address))
        return hex_to_decimal(result)

    def get_submission_ipfs_cid(self) -> str:
        """calls solidity method getSubmission()"""

        dataset_hash = self._call("getSubmission(uint32,address)",
                                  int_to_uint(self._challenge.number),
                                  address_to_uint(self._address))
        return hash_to_cid(dataset_hash)

    def download_submission_file(self, force=False, verbose=True) -> Path:
        """utility method to read the submission file cid and download the submission file in the challenge directory"""

        submission_ipfs_cid = self.get_submission_ipfs_cid()
        challenge_dir = self._challenge.get_challenge_dir()
        return download_ipfs_file(submission_ipfs_cid, challenge_dir, prefix="submission-", suffix=".zip", force=force, verbose=verbose)

    def get_challenge_reward(self) -> Decimal:
        """calls solidity method getChallengeRewards()"""

        result = self._call("getChallengeRewards(uint32,address)",
                            int_to_uint(self._challenge.number),
                            address_to_uint(self._address))
        return hex_to_decimal(result)

    def get_competition_reward(self) -> Decimal:
        """calls solidity method getTournamentRewards()"""

        result = self._call("getTournamentRewards(uint32,address)",
                            int_to_uint(self._challenge.number),
                            address_to_uint(self._address))
        return hex_to_decimal(result)

    def get_challenge_score(self) -> Decimal:
        """calls solidity method getChallengeScores()"""

        result = self._call("getChallengeScores(uint32,address)",
                            int_to_uint(self._challenge.number),
                            address_to_uint(self._address))
        return hex_to_decimal(result)

    def get_competition_score(self) -> Decimal:
        """calls solidity method getTournamentScores()"""

        result = self._call("getTournamentScores(uint32,address)",
                            int_to_uint(self._challenge.number),
                            address_to_uint(self._address))
        return hex_to_decimal(result)

    def _call(self, fn_signature: str, *args):
        return self._challenge.call(fn_signature, *args)


def get_competition_address(competition_name: str) -> str:
    data = encode_string(competition_name)
    params = [{"to": REGISTRY_ADDRESS, "data": data}, "latest"]
    return f"0x{network_read(params)[-40:]}"


def call(address: str, fn_signature: str, *args):
    fn_id = str_to_fn_id(fn_signature)
    data = fn_id
    for arg in args:
        data = data + arg
    params = [{"to": address, "data": data}, "latest"]
    return network_read(params)


def scan(from_block: int, to_block: int, address: str):
    params = [{"fromBlock": hex(from_block), "toBlock": hex(to_block), "address": address}]
    method = "eth_getLogs"
    return network_read(params, method)


def get_latest_block() -> int:
    params = ["latest", False]
    method = "eth_getBlockByNumber"
    return hex_to_int(network_read(params, method)["number"])


def network_read(params: [Any], method="eth_call") -> str:
    payload = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
    headers = {"Content-Type": "application/json"}

    retries = 0
    while retries < MAX_RETRIES:
        r = requests.post(POLYGON_GATEWAY, headers=headers, json=payload)
        if r.ok:
            return r.json()["result"]
    if retries >= MAX_RETRIES:
        assert False, "network read exceeded max retries. Please try again later"


# TODO: don't use absolute fn_id
def encode_string(param: str, fn_id="0x5d58ebc1") -> str:
    line2 = "0" * 62 + "20"
    line3 = hex(len(param))[2:]
    line3 = "0" * (64 - len(line3)) + line3
    param_line = ""
    for i in range(0, len(param), 32):
        chunk = param[i:i + 32]
        utf8_encoded = chunk.encode("utf-8").hex()
        padding = "0" * (64 - len(utf8_encoded))
        param_line += utf8_encoded + padding
    return fn_id + line2 + line3 + param_line


def hash_to_cid(hash_id: str) -> str:
    if hash_id[:2] == "0x":
        hash_id = hash_id[2:]
    hash_id = "1220" + str(hash_id)
    hash_id = int(hash_id, 16)
    return base58.b58encode_int(hash_id).decode("utf-8")


def str_to_fn_id(fn_signature: str, full_sig=False) -> str:
    hashed_string = keccak_256(fn_signature.encode("utf-8")).digest().hex()
    if not full_sig:
        hashed_string = hashed_string[:8]
    return f"0x{hashed_string}"


def int_to_uint(n: int) -> str:
    uint = hex(n)[2:]
    return "0" * (64 - len(uint)) + uint


def address_to_uint(address: str) -> str:
    return "0" * 24 + address


def hex_to_int(hex: str) -> int:
    return int(hex[2:], 16)


def hex_to_decimal(hex: str) -> Decimal:
    return Decimal(hex_to_int(hex)) / Decimal("1e18")


def download_ipfs_file(cid: str, target_dir: Path,
                       verbose=True, force=False, filename=None, prefix=None, suffix=None) -> Path:
    if filename is None:
        filename = cid
    if prefix is not None:
        filename = prefix + filename
    if suffix is not None:
        filename = filename + suffix

    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded_file = target_dir.joinpath(cid)
    target_file = target_dir.joinpath(filename)
    if not target_file.exists() or force:
        if verbose:
            print(f"downloading file {cid}")

        client = ipfshttpclient.connect(addr=IPFS_GATEWAY)
        client.get(cid, target_dir)
        if downloaded_file != target_file:
            downloaded_file.rename(target_file)

    return target_file
