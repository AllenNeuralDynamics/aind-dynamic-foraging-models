"""Testing logistic regression model"""

import os
import unittest
import zipfile

import numpy as np
import requests
from pynwb import NWBHDF5IO

from aind_dynamic_foraging_models.logistic_regression import (
    fit_logistic_regression,
    plot_logistic_regression,
)


# Start a new test case
class TestLogistic(unittest.TestCase):
    """Testing logistic regression model"""

    def setUp(self):
        """Set up the test case by downloading an example NWB file and extracting the data."""
        url = "https://github.com/user-attachments/files/16698772/example_nwb_1.zip"
        nwb_name = "703548_2024-03-20_10-47-42.nwb"
        example_data_path = "data/"
        zip_path = example_data_path + "example_nwb_1.zip"
        extract_path = example_data_path + "example_nwb_1/"

        os.makedirs(example_data_path, exist_ok=True)

        response = requests.get(url)
        with open(zip_path, "wb") as file:
            file.write(response.content)

        # Unzip the file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        # Load the NWB file
        io = NWBHDF5IO(extract_path + nwb_name, mode="r")
        nwb = io.read()
        df_trial = nwb.trials.to_dataframe()

        # Turn to 0 and 1 coding (assuming all trials are non-autowater trials)
        choice_history = df_trial["animal_response"].values
        choice_history[choice_history == 2] = np.nan
        reward_history = (
            (df_trial["rewarded_historyL"] + df_trial["rewarded_historyR"]).astype(int).values
        )

        self.choice_history = choice_history
        self.reward_history = reward_history

    def test_logistic_regression(self):
        """Testing logistic regression model"""

        # -- Call logistic regression --
        dict_logistic_result = fit_logistic_regression(
            self.choice_history,
            self.reward_history,
            logistic_model="Su2022",
            n_trial_back=15,
            selected_trial_idx=None,
            solver="liblinear",
            penalty="l2",
            Cs=10,
            cv=10,
            n_jobs_cross_validation=-1,
            n_bootstrap_iters=1000,
            n_bootstrap_samplesize=None,
        )

        # -- Check the result --
        df_beta = dict_logistic_result["df_beta"]

        np.testing.assert_almost_equal(
            df_beta.cross_validation[:10].to_numpy(),
            np.array(
                [
                    -0.14618642,
                    0.06559765,
                    0.13649798,
                    -0.10842585,
                    0.28067104,
                    0.29741248,
                    -0.04829112,
                    0.0008956,
                    0.21110653,
                    0.1979053,
                ]
            ),
            decimal=3,
        )
        np.testing.assert_almost_equal(
            dict_logistic_result["df_beta_exp_fit"].iloc[0, :].to_numpy(),
            np.array([1.29269355, 0.32109687, 2.6384484, 0.75849041]),
            decimal=3,
        )

        # -- Check plotting --
        ax = plot_logistic_regression(dict_logistic_result)
        ax.get_figure().savefig("tests/results/test_logistic_regression.png")
        self.assertIsNotNone(ax)


if __name__ == "__main__":
    unittest.main(verbosity=2)
