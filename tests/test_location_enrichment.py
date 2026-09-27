from wildlife_csi.location_enrichment import JsonlCheckpoint, observation_country


def test_observation_country_uses_only_country_level_public_places():
    observation = {"place_ids": [1, 7, 9], "geoprivacy": None}
    places = {
        1: {"found": True, "name": "United States", "admin_level": 0, "place_type": 12},
        7: {"found": True, "name": "Virginia", "admin_level": 10, "place_type": 8},
        9: {"found": False, "name": None, "admin_level": None, "place_type": None},
    }
    assert observation_country(observation, places) == {
        "country": "United States",
        "level": "country",
        "basis": "observation_public_place",
        "is_observation_location": True,
        "source": "iNaturalist",
        "source_place_id": 1,
        "geoprivacy": "open",
    }


def test_observation_country_keeps_conflicts_unresolved():
    observation = {"place_ids": [1, 2], "geoprivacy": "obscured"}
    places = {
        1: {"found": True, "name": "United States", "admin_level": 0, "place_type": 12},
        2: {"found": True, "name": "Canada", "admin_level": 0, "place_type": 12},
    }
    result = observation_country(observation, places)
    assert result["country"] is None
    assert result["is_observation_location"] is False
    assert result["candidate_countries"] == ["Canada", "United States"]
    assert result["geoprivacy"] == "obscured"


def test_observation_country_flags_explicit_place_guess_disagreement():
    observation = {"place_ids": [1], "place_guess": "Zimbabwe", "geoprivacy": None}
    places = {
        1: {"found": True, "name": "Botswana", "admin_level": 0, "place_type": 12},
        2: {"found": True, "name": "Zimbabwe", "admin_level": 0, "place_type": 12},
    }
    result = observation_country(observation, places)
    assert result["basis"] == "observation_public_place_with_text_conflict"
    assert result["candidate_countries"] == ["Botswana", "Zimbabwe"]
    assert result["country"] == "Botswana"
    assert result["is_observation_location"] is True


def test_taiwan_and_puerto_rico_public_places():
    places = {
        7887: {"found": True, "name": "Taiwan", "admin_level": 0, "place_type": None},
        6848: {"found": True, "name": "Puerto Rico", "admin_level": 0, "place_type": 101},
    }
    assert observation_country({"place_ids": [7887]}, places)["country"] == "Taiwan"
    puerto_rico = observation_country({"place_ids": [6848]}, places)
    assert puerto_rico["country"] == "United States"
    assert puerto_rico["region"] == "Puerto Rico"


def test_jsonl_checkpoint_survives_restart(tmp_path):
    path = tmp_path / "observations.jsonl"
    cache = JsonlCheckpoint(path, "observation_id")
    cache.put({"observation_id": "1", "place_ids": [7]})
    assert JsonlCheckpoint(path, "observation_id").rows["1"]["place_ids"] == [7]
