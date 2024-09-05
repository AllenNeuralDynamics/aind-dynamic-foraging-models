"""Test module for q_learning_params.py"""

import unittest

from pydantic import ValidationError

from aind_dynamic_foraging_models.generative_model.params.forager_loss_counting_params import (
    generate_pydantic_loss_counting_params,
)
from aind_dynamic_foraging_models.generative_model.params.forager_q_learning_params import (
    generate_pydantic_q_learning_params,
)
from aind_dynamic_foraging_models.generative_model.params.util import get_params_options


class TestParamsSimpleQ(unittest.TestCase):
    """Test generating Pydantic models for Q-learning agent parameters"""

    def test_generate_models_RW1972(self):
        """Test generating pydantic models for RW1972 agent"""
        # Create Pydantic models
        ParamsModel, FittingBoundsModel = generate_pydantic_q_learning_params(
            number_of_learning_rate=1,
            number_of_forget_rate=0,
            choice_kernel="none",
            action_selection="epsilon-greedy",
        )
        expected_fields = ["learn_rate", "epsilon", "biasL"]

        self.check_fields(ParamsModel, FittingBoundsModel, expected_fields)
        self.check_validation(ParamsModel, FittingBoundsModel)

    def test_generate_models_Bari2019(self):
        """Test generating pydantic models for Bari2019 agent"""
        # Create Pydantic models
        ParamsModel, FittingBoundsModel = generate_pydantic_q_learning_params(
            number_of_learning_rate=1,
            number_of_forget_rate=1,
            choice_kernel="one_step",
            action_selection="softmax",
        )
        expected_fields = [
            "learn_rate",
            "forget_rate_unchosen",
            "choice_kernel_step_size",
            "choice_kernel_relative_weight",
            "softmax_inverse_temperature",
            "biasL",
        ]

        self.check_fields(ParamsModel, FittingBoundsModel, expected_fields)
        self.check_validation(ParamsModel, FittingBoundsModel)

        # Make sure choice_kernel_step_size is fixed to 1.0
        self.assertEqual(ParamsModel.model_fields["choice_kernel_step_size"].default, 1.0)
        with self.assertRaises(ValidationError):
            ParamsModel(choice_kernel_step_size=0.5)

        # Show params options
        params_options = get_params_options(ParamsModel)
        print("\n", params_options)

    def test_generate_models_Hattori2019(self):
        """Test generating pydantic models for Hattori2019 agent"""
        # Create Pydantic models
        ParamsModel, FittingBoundsModel = generate_pydantic_q_learning_params(
            number_of_learning_rate=2,
            number_of_forget_rate=1,
            choice_kernel="none",
            action_selection="softmax",
        )
        expected_fields = [
            "learn_rate_rew",
            "learn_rate_unrew",
            "forget_rate_unchosen",
            "softmax_inverse_temperature",
            "biasL",
        ]

        self.check_fields(ParamsModel, FittingBoundsModel, expected_fields)
        self.check_validation(ParamsModel, FittingBoundsModel)

    def check_fields(self, ParamsModel, FittingBoundsModel, expected_fields):
        """Check fields of Pydantic models"""
        self.assertEqual(set(ParamsModel.model_fields.keys()), set(expected_fields))
        self.assertEqual(set(FittingBoundsModel.model_fields.keys()), set(expected_fields))

    def check_validation(self, ParamsModel, FittingBoundsModel):
        """Check validation of Pydantic models"""
        with self.assertRaises(ValidationError):
            ParamsModel(learn_rate=1.1)
        with self.assertRaises(ValidationError):
            FittingBoundsModel(learn_rate=(1.1, 1.0))


class TestParamsLossCounting(unittest.TestCase):
    """Test generating Pydantic models for loss counting agent parameters"""

    def test_generate_models_LossCounting(self):
        """Test generating pydantic models for loss counting agent"""
        # Create Pydantic models
        ParamsModel, FittingBoundsModel = generate_pydantic_loss_counting_params()
        expected_fields = ["loss_count_threshold_mean", "loss_count_threshold_std", "biasL"]

        self.check_fields(ParamsModel, FittingBoundsModel, expected_fields)
        self.check_validation(ParamsModel, FittingBoundsModel)

    def check_fields(self, ParamsModel, FittingBoundsModel, expected_fields):
        """Check fields of Pydantic models"""
        self.assertEqual(set(ParamsModel.model_fields.keys()), set(expected_fields))
        self.assertEqual(set(FittingBoundsModel.model_fields.keys()), set(expected_fields))

    def check_validation(self, ParamsModel, FittingBoundsModel):
        """Check validation of Pydantic models"""
        with self.assertRaises(ValidationError):
            ParamsModel(loss_count_threshold_mean=-0.1)
        with self.assertRaises(ValidationError):
            FittingBoundsModel(loss_count_threshold_std=(1.0, 0.9))


if __name__ == "__main__":
    unittest.main(verbosity=2)
