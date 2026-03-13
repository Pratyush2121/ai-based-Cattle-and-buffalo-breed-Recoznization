def test_train_simple():
    from src.train import train

    model = train()
    assert "coef" in model
    assert model["coef"] == 2.0
