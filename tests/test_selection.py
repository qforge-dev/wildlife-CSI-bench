from PIL import Image

from wildlife_csi.animalclue import REPOS
from wildlife_csi.builder import SuiteBuilder, trace_box
from wildlife_csi.selection import image_quality, quality_rejection, select_diverse


def candidate(species: int, obs: str, score: float) -> dict:
    return {
        "correct_taxon_id": species,
        "image_sha256": f"{int(obs):064x}",
        "source": {"observation_id": obs, "quality": {"score": score}},
    }


def test_footprints_use_yolo_boxes_and_pick_largest_trace():
    class_id, box, count = trace_box(
        "footprint",
        "7 0.5 0.5 0.1 0.1\n7 0.5 0.5 0.4 0.5\n",
        1000,
        800,
    )
    assert (class_id, count) == (7, 2)
    assert box[2] - box[0] > 400
    assert box[3] - box[1] > 400


def test_selection_balances_species_then_prefers_quality():
    pool = [
        candidate(1, "1", 0.9),
        candidate(1, "2", 0.8),
        candidate(1, "3", 0.7),
        candidate(2, "4", 0.6),
        candidate(3, "5", 0.5),
    ]
    chosen = select_diverse(pool, 4, 42, "egg")
    assert {t["correct_taxon_id"] for t in chosen} == {1, 2, 3}
    assert {t["source"]["observation_id"] for t in chosen} == {"1", "2", "4", "5"}
    assert chosen == select_diverse(pool, 4, 42, "egg")


def test_segmentations_use_largest_annotated_trace():
    class_id, box, count = trace_box(
        "bone",
        "9 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n9 0.3 0.3 0.8 0.3 0.8 0.8 0.3 0.8\n",
        1000,
        1000,
    )
    assert (class_id, count) == (9, 2)
    assert box[2] - box[0] > 500


def test_quality_prefers_larger_sharper_image():
    small = Image.new("RGB", (400, 400), "gray")
    large = Image.new("RGB", (1000, 1000), "gray")
    assert image_quality(large)["score"] > image_quality(small)["score"]
    assert (
        quality_rejection(
            "footprint",
            {
                "native_crop_short_side": 700,
                "contrast": 40,
                "detail": 1,
                "score": 0.8,
            },
        )
        == "low_detail_crop"
    )


def test_builder_chooses_best_photo_within_observation(tmp_path):
    paths = []
    for index, size in enumerate((500, 1100)):
        path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (size, size), "gray").save(path)
        paths.append(path)
    label = tmp_path / "label.txt"
    label.write_text("9 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n")

    class Repo:
        def download(self, spec, path):
            return label if path.endswith(".txt") else paths[int(path.split("_")[1].split(".")[0])]

    observation = {
        "id": 123,
        "quality_grade": "research",
        "taxon": {"id": 1},
        "photos": [
            {"id": 10, "license_code": "cc-by", "attribution": "A"},
            {"id": 11, "license_code": "cc-by", "attribution": "A"},
        ],
    }
    taxon = {
        "id": 1,
        "name": "Vulpes vulpes",
        "rank": "species",
        "ancestors": [
            {"id": 3, "name": "Canidae", "rank": "family"},
            {"id": 2, "name": "Vulpes", "rank": "genus"},
        ],
    }
    builder = SuiteBuilder(
        Repo(),
        None,
        quality_scorer=lambda img: {
            "native_crop_short_side": min(img.size),
            "contrast": 50,
            "detail": 10,
            "score": min(img.size) / 1000,
        },
        location_resolver=lambda observation, source: {
            "country": "United States", "level": "country", "basis": "observation_public_place",
            "is_observation_location": True, "source": "iNaturalist", "source_place_id": 1,
            "geoprivacy": "open", "observation_id": source["observation_id"],
        },
    )
    task, _ = builder._prepare_observation(
        "egg",
        REPOS["egg"],
        "123",
        [("123_0.jpg", "123_0.txt"), ("123_1.jpg", "123_1.txt")],
        observation,
        taxon,
    )
    assert task["source"]["image_path"] == "123_1.jpg"
    assert task["source"]["observation_photo_candidates"] == 2
