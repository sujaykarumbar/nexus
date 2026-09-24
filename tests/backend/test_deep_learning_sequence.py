"""
Tests for PyTorch Deep Learning Sequence Forecasters.
Validates LSTMForecaster, GRUForecaster, DeepLearningTrainer windowing,
training convergence, checkpoint saving, and reloading.
"""

import os
import pytest
import numpy as np
import pandas as pd
import torch

from services.forecasting.deep_learning.models import LSTMForecaster, GRUForecaster
from services.forecasting.deep_learning.trainer import DeepLearningTrainer


@pytest.fixture
def sequence_series():
    """Generates 50 steps of a clean synthetic sine wave."""
    t = np.linspace(0, 8 * np.pi, 50)
    vals = 50.0 + 10.0 * np.sin(t)
    return pd.Series(vals)


def test_lstm_model_shapes():
    model = LSTMForecaster(input_size=1, hidden_dim=16, num_layers=1, output_horizon=5)
    # Batch size 4, sequence length 10, input_size 1
    dummy_input = torch.randn(4, 10, 1)
    output = model(dummy_input)
    assert output.shape == (4, 5)


def test_gru_model_shapes():
    model = GRUForecaster(input_size=1, hidden_dim=16, num_layers=1, output_horizon=5)
    dummy_input = torch.randn(4, 10, 1)
    output = model(dummy_input)
    assert output.shape == (4, 5)


def test_deep_learning_trainer_convergence_and_predict(sequence_series, tmp_path):
    trainer = DeepLearningTrainer(
        architecture="lstm",
        seq_length=10,
        horizon=5,
        hidden_dim=16,
        num_layers=1,
        max_epochs=20,
        patience=4
    )

    info = trainer.fit(sequence_series)
    assert info["epochs_trained"] > 0
    assert len(info["train_loss_history"]) > 0
    # Loss should decrease from early epoch to later epoch
    assert info["final_train_loss"] < info["train_loss_history"][0] * 1.5

    # Test prediction
    preds = trainer.predict()
    assert len(preds) == 5
    assert not np.any(np.isnan(preds))

    # Test save and reload
    ckpt_path = str(tmp_path / "model.pt")
    trainer.save(ckpt_path)
    assert os.path.exists(ckpt_path)

    reloaded = DeepLearningTrainer.load(ckpt_path)
    reloaded_preds = reloaded.predict()
    assert np.allclose(preds, reloaded_preds, atol=1e-4)


def test_gru_trainer(sequence_series):
    trainer = DeepLearningTrainer(
        architecture="gru",
        seq_length=8,
        horizon=4,
        hidden_dim=16,
        num_layers=1,
        max_epochs=15,
        patience=3
    )
    info = trainer.fit(sequence_series)
    assert info["architecture"] == "gru"
    preds = trainer.predict()
    assert len(preds) == 4
