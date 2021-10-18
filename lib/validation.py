import shutil
import os
import numpy as np
import pandas as pd

import lib.scoring as scoring

from decimal import Decimal
from pathlib import Path
from Crypto.Cipher import PKCS1_OAEP, AES
from Crypto.PublicKey import RSA

from lib.blockchain import Challenge, Participant


def compute_error(participant: Participant, force=False) -> float:
    """computes the RMSE of a participant's prediction w.r.t. correct values"""

    challenge = participant.challenge
    competition = challenge.competition

    challenge_scores_file = challenge.get_challenge_dir().joinpath("_challenge_scores.csv")
    if challenge_scores_file.exists() and not force:
        df = pd.read_csv(challenge_scores_file)
        return df.error.loc[df.address == participant.address].values[0]

    # check if requested challenge is not the last
    if challenge.get_phase() != 4:
        raise IndexError(f"challenge {challenge.number} is not closed")

    if challenge.number == competition.get_latest_challenge_number():
        raise IndexError(f"no challenge after challenge {challenge.number}")

    next_challenge = competition.get_challenge(challenge.number + 1)
    if next_challenge.get_phase() == 0:
        raise IndexError(f"dataset not available for challenge {challenge.number}")

    # get requested assets and challenge final values
    requested_assets = get_requested_assets(challenge, force=force)
    values = get_values(next_challenge, force=force)
    valued_assets = [asset for asset, value in values]

    # if the participant did not submit a prediction return NaN
    submitters = challenge.get_all_submitter_addresses()  # TODO: a set would be better
    if participant.address not in submitters:
        return np.nan

    # get the predictions, and if not valid return NaN
    predictions_pairs = get_predictions(participant)
    if not scoring.validate_prediction(requested_assets, predictions_pairs):
        return np.nan

    # sort predictions values according to assets list
    predictions_by_asset = dict(predictions_pairs)
    predictions = [predictions_by_asset[asset] for asset in valued_assets]

    # return error computed from predictions and correct values
    return scoring.compute_challenge_error(challenge.number, predictions, [value for _, value in values])


def compute_challenge_scores(challenge: Challenge, force=False) -> {str: float}:
    """compute the list of all participant challenge scores for a given challenge"""

    challenge_scores_file = challenge.get_challenge_dir().joinpath("_challenge_scores.csv")
    if challenge_scores_file.exists() and not force:
        df = pd.read_csv(challenge_scores_file)
        return dict(zip(df.address, df.challenge_score))

    # read all submitters to the challenge
    participants = challenge.get_all_participants()

    # compute all participant errors
    errors = [compute_error(participant, force) for participant in participants]

    # compute the normalized rank of each submitter from the error
    challenge_scores = scoring.compute_challenge_scores(challenge.number, errors)

    # save errors and challenge scores to file
    df = pd.DataFrame()
    df["address"] = [participant.address for participant in participants]
    df["error"] = errors
    df["challenge_score"] = challenge_scores
    df.to_csv(challenge_scores_file, index=False)

    # return a dictionary with the challenge scores
    return {participant.address: score for participant, score in zip(participants, challenge_scores)}


def compute_challenge_score(participant: Participant, force=False) -> float:
    """computes the challenge score of a participant_address to a challenge"""

    challenge_scores = compute_challenge_scores(participant.challenge, force)
    return challenge_scores.get(participant.address, np.nan)


def compute_competition_score(participant: Participant, force=False) -> float:
    """computes the competition score of a participant_address to a challenge"""

    challenge = participant.challenge
    competition = challenge.competition

    # gets last 4 challenge scores of a submitter and compute the competition score
    challenge_scores = [compute_challenge_score(challenge.get_participant(participant.address), force)
                        for challenge in [competition.get_challenge(challenge_number)
                                          for challenge_number in range(max(1, challenge.number - 3),
                                                                        challenge.number + 1)]]
    return scoring.compute_competition_score(challenge.number, challenge_scores)


def compute_challenge_rewards(challenge: Challenge) -> {str, Decimal}:
    """computes the challenge rewards of challenge"""

    challenge_pool = challenge.get_challenge_pool()
    addresses = [staker.address for staker in challenge.get_all_participants()]
    scores_by_address = compute_challenge_scores(challenge)
    scores = [scores_by_address.get(address) for address in addresses]
    rewards = scoring.compute_challenge_rewards(challenge.number, scores, challenge_pool)
    return dict(zip(addresses, rewards))


def compute_competition_rewards(challenge: Challenge) -> {str, Decimal}:
    """computes the competition reward of challenge"""

    competition_pool = challenge.get_competition_pool()
    addresses = [staker.address for staker in challenge.get_all_participants()]
    scores = [compute_competition_score(participant) for participant in challenge.get_all_participants()]
    rewards = scoring.compute_competition_rewards(challenge.number, scores, competition_pool)
    return dict(zip(addresses, rewards))


def get_stakes(challenge: Challenge) -> {str, Decimal}:
    """returns the stakes of all participants to a challenge"""

    return {staker.address: staker.get_stake() for staker in challenge.get_all_participants()}


def compute_stake_rewards(challenge: Challenge) -> {str, Decimal}:
    """computes all stake rewards in a challenge"""

    stake_pool = challenge.get_stake_pool()
    stakers = challenge.get_all_participants()
    addresses = [staker.address for staker in stakers]
    stakes = [staker.get_stake() for staker in stakers]
    rewards = scoring.compute_stake_rewards(challenge.number, stakes, stake_pool)
    return dict(zip(addresses, rewards))


def get_predictions(participant: Participant) -> [(str, Decimal)]:
    """gets all predictions of a submitter to a challenge"""

    # get the submission cid and download the prediction file if needed
    submission_zip_file = participant.download_submission_file(verbose=True)

    # expand the file
    submission_dir = submission_zip_file.parent.joinpath(submission_zip_file.stem)
    shutil.unpack_archive(submission_zip_file, submission_dir)

    # decrypt the file
    encrypted_symmetric_key_file = submission_dir.joinpath("encrypted_symmetric_key.pem")
    private_key_file = participant.challenge.download_private_key_file(verbose=True)
    symmetric_key_file = submission_dir.joinpath("_symmetric_key.bin")
    _asymmetric_decrypt_file(encrypted_symmetric_key_file, private_key_file, symmetric_key_file)

    # decrypt and read the originator file and check if the originator is the submitter
    encrypted_originator_file = submission_dir.joinpath("originator.bin")
    originator_file = submission_dir.joinpath("_originator.txt")
    _decrypt_file(encrypted_originator_file, symmetric_key_file, originator_file)
    with open(originator_file, "r") as fin:
        originator_address = fin.read().strip()
    assert originator_address[2:].lower() == participant.address.lower()

    # check and decrypt the submission file
    encrypted_prediction_filenames = [filename for filename in os.listdir(submission_dir)
                                      if submission_dir.joinpath(filename).is_file()
                                      and submission_dir.joinpath(filename).match("*.bin")
                                      and filename not in ["originator.bin", "_symmetric_key.bin"]]
    assert len(encrypted_prediction_filenames) == 1

    encrypted_prediction_file = submission_dir.joinpath(encrypted_prediction_filenames[0])
    prediction_file = submission_dir.joinpath("_predictions.csv")
    _decrypt_file(encrypted_prediction_file, symmetric_key_file, prediction_file)

    # load the file and returns the list of pairs
    df = pd.read_csv(prediction_file, header=None)
    return df.to_records(index=False)


def get_requested_assets(challenge: Challenge, force=False) -> [str]:
    """gets assets to be predicted at the beginning of the challenge"""

    # download and unzip the dataset
    dataset_zip_file = challenge.download_dataset_file(force=force, verbose=True)
    dataset_dir = dataset_zip_file.parent.joinpath(dataset_zip_file.stem)
    if not dataset_dir.exists() or force:
        shutil.unpack_archive(dataset_zip_file, dataset_dir)

    # load the test dataset if needed in memory
    assets_dataset_file = dataset_dir.joinpath("_assets.csv")
    if not assets_dataset_file.exists() or force:
        validation_dataset_file = dataset_dir.joinpath("dataset/validation_dataset.csv")
        df = pd.read_csv(validation_dataset_file)
        df = df[["symbol"]]

        # save assets to file and return them
        df.to_csv(assets_dataset_file, index=False)
    else:
        # read assets from file
        df = pd.read_csv(assets_dataset_file)

    # convert dataframe array of values
    return df.symbol.values


def get_values(challenge: Challenge, force=False) -> [(str, Decimal)]:
    """gets correct values of predictions at the end of a challenge"""

    # download and unzip the dataset
    dataset_zip_file = challenge.download_dataset_file(force=force, verbose=True)
    dataset_dir = dataset_zip_file.parent.joinpath(dataset_zip_file.stem)
    if not dataset_dir.exists() or force:
        shutil.unpack_archive(dataset_zip_file, dataset_dir)

    # load the train dataset if needed in memory and extract the values from last week
    values_dataset_file = dataset_dir.joinpath("_values.csv")
    if not values_dataset_file.exists() or force:
        train_dataset_file = dataset_dir.joinpath("dataset/train_dataset.csv")
        df = pd.read_csv(train_dataset_file)
        max_date = max(df.date)
        df = df[["date", "symbol", "target"]]
        values = df[df.date == max_date]

        # save assets to file
        values.to_csv(values_dataset_file, index=False)
    else:
        # read assets from file
        values = pd.read_csv(values_dataset_file)

    # convert dataframe to array of tuples
    return values[["symbol", "target"]].to_records(index=False)


def _asymmetric_decrypt_file(input_file: Path, private_key_file: Path, output_file: Path) -> None:

    # read private key
    with open(private_key_file, "rb") as fin:
        private_key_data = fin.read()
    private_key = RSA.import_key(private_key_data)

    # read encrypted data
    with open(input_file, "rb") as dec_f:
        ciphertext = dec_f.read()

    # decrypt and save to file the encrypted symmetric key
    cipher = PKCS1_OAEP.new(private_key)
    cleartext = cipher.decrypt(ciphertext)
    with open(output_file, "wb") as dec_f:
        dec_f.write(cleartext)


def _decrypt_file(input_file: Path, symmetric_key_file: Path, output_file: Path) -> None:

    # read key
    with open(symmetric_key_file, "rb") as kin:
        symmetric_key = kin.read()

    # read data and split into nonce, cyphertext and tag
    with open(input_file, "rb") as fin:
        encrypted_data = fin.read()
    nonce = encrypted_data[:16]
    ciphertext = encrypted_data[16:-16]
    tag = encrypted_data[-16:]

    # decrypt and save to file
    cipher = AES.new(symmetric_key, AES.MODE_GCM, nonce)
    cleartext = cipher.decrypt_and_verify(ciphertext, tag)
    with open(output_file, "wb") as fout:
        fout.write(cleartext)


