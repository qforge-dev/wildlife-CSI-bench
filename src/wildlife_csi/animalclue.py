"""AnimalClue source access: pinned revisions, file layout, YOLO parsing and crop.

The Hugging Face datasets require authorized access. Repository bytes are
downloaded only after the account has been granted that access.
"""

from __future__ import annotations

from pathlib import Path

REPOS = {
    "footprint": {
        "repo": "risashinoda/footprint_yolo",
        "revision": "600ecddca54a8513a9a7f79758ec7197b85e5bf3",
        "info": "info_footprint.csv",
    },
    "feces": {
        "repo": "risashinoda/feces_yolo",
        "revision": "140f14cda1e4ea2b18748b76698c2f45318169a9",
        "info": "info_feces.csv",
    },
    "egg": {
        "repo": "risashinoda/egg_yolo",
        "revision": "bf7bfdf85bb5a565e089afd34f4b2f02cd82b6d8",
        "info": "info_egg.csv",
    },
    "bone": {
        "repo": "risashinoda/bone_yolo",
        "revision": "a0f7b8561fd234af3d91e69b7da5abaa9a12330a",
        "info": "info_bone.csv",
    },
    "feather": {
        "repo": "risashinoda/feather_yolo",
        "revision": "e0194b1823d02bf549b90ed1c711885c7e62b477",
        "info": "info_feather.csv",
    },
}

# Expected test-split prefixes (bone layout verified via file listing;
# others share species/{train,valid,test}/ convention).
TEST_PREFIX = "species/test/"


class AccessPending(RuntimeError):
    pass


def _token() -> str | None:
    try:
        tok = Path.home().joinpath(".cache/huggingface/token").read_text().strip()
        return tok or None
    except OSError:
        return None


def require_token() -> str:
    tok = _token()
    if not tok:
        raise AccessPending("no HF token: run `hf auth login` first")
    return tok


def check_access() -> dict:
    """List-only audit: works without author approval. Returns status per repo."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError

    api = HfApi(token=_token())
    out = {}
    for clue, spec in REPOS.items():
        row: dict = {"repo": spec["repo"], "pinned": spec["revision"][:12]}
        try:
            info = api.repo_info(spec["repo"], repo_type="dataset")
            row["live_rev"] = info.sha or ""
            row["rev_match"] = (info.sha or "") == spec["revision"]
            siblings = [s.rfilename for s in (info.siblings or [])]
            row["n_files"] = len(siblings)
            row["has_test"] = any(s.startswith(TEST_PREFIX) for s in siblings)
        except Exception as e:
            row["error"] = str(e)[:200]
            out[clue] = row
            continue
        # Byte probe: one tiny metadata file. 403 here == pending author review.
        try:
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                spec["repo"],
                spec["info"],
                repo_type="dataset",
                revision=spec["revision"],
                token=_token(),
            )
            row["bytes"] = "ok"
        except GatedRepoError:
            row["bytes"] = "pending-author-review"
        except Exception as e:
            row["bytes"] = f"error: {str(e)[:120]}"
        out[clue] = row
    return out


def require_access(clues: list[str]) -> None:
    """Fail before a suite build downloads images if any selected source is gated."""
    unknown = set(clues) - REPOS.keys()
    if unknown:
        raise ValueError(f"unknown clues: {sorted(unknown)}")
    status = check_access()
    blocked = [
        f"{clue} ({REPOS[clue]['repo']}): {status.get(clue, {}).get('bytes', status.get(clue, {}).get('error', 'unavailable'))}"
        for clue in clues
        if status.get(clue, {}).get("bytes") != "ok"
    ]
    if blocked:
        raise AccessPending("AnimalClue access required before building:\n" + "\n".join(blocked))


def observation_id(filename: str) -> str:
    """Filenames lead with <observation_id>_<index>.... e.g. 100159879_0_jpeg..."""
    return Path(filename).name.split("_")[0]


def parse_yolo_label(text: str) -> list[tuple[int, float, float, float, float]]:
    """Parse YOLO-bbox lines -> [(class_id, cx, cy, w, h)] normalized. Skips blanks."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        out.append(
            (int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        )
    return out


def parse_seg_label(text: str) -> list[tuple[int, list[float]]]:
    """Parse YOLO-seg lines -> [(class_id, [x0,y0,x1,y1,...])] normalized."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        out.append((int(parts[0]), [float(v) for v in parts[1:]]))
    return out


def seg_bbox(
    xs: list[float], img_w: int, img_h: int, ys: list[float] | None = None, margin: float = 0.15
) -> tuple[int, int, int, int]:
    """Polygon coords (normalized, alternating x y) -> pixel crop with margin."""
    if ys is None:
        xs, ys = xs[0::2], xs[1::2]
    x0, x1 = min(xs) * img_w, max(xs) * img_w
    y0, y1 = min(ys) * img_h, max(ys) * img_h
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    return (
        max(0, int(x0 - bw * margin)),
        max(0, int(y0 - bh * margin)),
        min(img_w, int(x1 + bw * margin)),
        min(img_h, int(y1 + bh * margin)),
    )


def crop_box(
    img_w: int, img_h: int, cx: float, cy: float, w: float, h: float, margin: float = 0.15
) -> tuple[int, int, int, int]:
    """YOLO normalized box -> pixel crop with fractional margin, clamped."""
    x0 = (cx - w / 2) * img_w
    y0 = (cy - h / 2) * img_h
    x1 = (cx + w / 2) * img_w
    y1 = (cy + h / 2) * img_h
    bw, bh = x1 - x0, y1 - y0
    x0 -= bw * margin
    y0 -= bh * margin
    x1 += bw * margin
    y1 += bh * margin
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img_w, x1), min(img_h, y1)
    return int(x0), int(y0), int(x1), int(y1)
