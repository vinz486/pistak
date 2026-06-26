import os
import json
import pytest
from pistak.config import ConfigManager

def test_config_manager_defaults(tmp_path):
    config_file = tmp_path / "settings.json"
    models_dir = tmp_path / "models"
    cm = ConfigManager(config_file=str(config_file), models_dir=str(models_dir))
    
    assert cm.settings["device"] == "CPU"
    assert cm.settings["port"] == "1234"

def test_config_save_and_load(tmp_path):
    config_file = tmp_path / "settings.json"
    models_dir = tmp_path / "models"
    cm1 = ConfigManager(config_file=str(config_file), models_dir=str(models_dir))
    
    cm1.settings["device"] = "GPU"
    cm1.settings["port"] = "9000"
    cm1.save_settings()
    
    assert config_file.exists()
    
    cm2 = ConfigManager(config_file=str(config_file), models_dir=str(models_dir))
    cm2.load_settings()
    
    assert cm2.settings["device"] == "GPU"
    assert cm2.settings["port"] == "9000"
    assert models_dir.exists()

def test_get_local_models(tmp_path):
    config_file = tmp_path / "settings.json"
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    
    (models_dir / "model_A").mkdir()
    (models_dir / "model_B").mkdir()
    (models_dir / "file.txt").touch()
    
    cm = ConfigManager(config_file=str(config_file), models_dir=str(models_dir))
    models = cm.get_local_models()
    
    model_names = [m[0] for m in models]
    assert "model_A" in model_names
    assert "model_B" in model_names
    assert "file.txt" not in model_names
