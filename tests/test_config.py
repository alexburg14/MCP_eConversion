"""Config loads from config.toml and is immutable."""
import dataclasses

import pytest

from config import get_config


def test_config_loads_required_sections():
    c = get_config()
    assert c.cluster.name
    assert c.cluster.display_name
    assert c.llm.base_url.startswith("http")
    assert len(c.llm.models) >= 1


def test_default_model_is_selectable():
    # The UI selects the default in the model list; a default outside the list
    # would silently fall back and mislead operators.
    c = get_config()
    assert c.llm.default_model in c.llm.models


def test_config_is_frozen():
    c = get_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.cluster.name = "mutated"
