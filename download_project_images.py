"""
Download a starter set of content/style images for proposal-scale experiments.

Content images are natural photos from Wikimedia Commons.
Style images are famous paintings mapped to artist slugs used by deception score.
"""

import argparse
import csv
from pathlib import Path
import requests


CONTENT_IMAGES = [
    (
        "golden_gate.jpg",
        "https://commons.wikimedia.org/wiki/Special:FilePath/GoldenGateBridge-001.jpg",
    ),
    (
        "mountain_lake.jpg",
        "https://commons.wikimedia.org/wiki/Special:FilePath/Moraine_Lake_17092005.JPG",
    ),
    (
        "city_night.jpg",
        "https://commons.wikimedia.org/wiki/Special:FilePath/Lower_Manhattan_from_Jersey_City_November_2014_panorama_3.jpg",
    ),
    (
        "forest_path.jpg",
        "https://commons.wikimedia.org/wiki/Special:FilePath/Forest_path,_Bialowieza_Forest,_Poland.jpg",
    ),
    (
        "desert.jpg",
        "https://commons.wikimedia.org/wiki/Special:FilePath/Sand_Dunes_in_Sahara.jpg",
    ),
    (
        "seaside.jpg",
        "https://commons.wikimedia.org/wiki/Special:FilePath/Pacific_Ocean_-_panoramio.jpg",
    ),
    (
        "village.jpg",
        "https://commons.wikimedia.org/wiki/Special:FilePath/Hallstatt_-_Zentrum.jpg",
    ),
    (
        "river_bridge.jpg",
        "https://commons.wikimedia.org/wiki/Special:FilePath/Charles_Bridge_Prague_Czech_Republic.jpg",
    ),
    (
        "field.jpg",
        "https://commons.wikimedia.org/wiki/Special:FilePath/Wheatfield.jpg",
    ),
    (
        "waterfall.jpg",
        "https://commons.wikimedia.org/wiki/Special:FilePath/Seljalandsfoss,_Iceland_(2).jpg",
    ),
]


STYLE_IMAGES = [
    (
        "starry_night.jpg",
        "vincent-van-gogh",
        "https://commons.wikimedia.org/wiki/Special:FilePath/The_Starry_Night.jpg",
    ),
    (
        "water_lilies.jpg",
        "claude-monet",
        "https://commons.wikimedia.org/wiki/Special:FilePath/Claude_Monet_-_Nympheas_-_Google_Art_Project.jpg",
    ),
    (
        "mont_sainte_victoire.jpg",
        "paul-cezanne",
        "https://commons.wikimedia.org/wiki/Special:FilePath/Paul_C%C3%A9zanne_150.jpg",
    ),
    (
        "yellow_christ.jpg",
        "paul-gauguin",
        "https://commons.wikimedia.org/wiki/Special:FilePath/Paul_Gauguin_047.jpg",
    ),
    (
        "composition_viii.jpg",
        "wassily-kandinsky",
        "https://commons.wikimedia.org/wiki/Special:FilePath/Vassily_Kandinsky,_1923_-_Composition_8.jpg",
    ),
]


def _download(url: str, dst: Path) -> None:
    if dst.exists():
        print(f"Skip existing: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dst}")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    dst.write_bytes(resp.content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download starter content/style images.")
    parser.add_argument("--content-dir", default="images/content")
    parser.add_argument("--style-dir", default="images/style")
    parser.add_argument("--manifest-path", default="images/style_manifest.csv")
    args = parser.parse_args()

    content_dir = Path(args.content_dir)
    style_dir = Path(args.style_dir)
    manifest_path = Path(args.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    for filename, url in CONTENT_IMAGES:
        try:
            _download(url, content_dir / filename)
        except Exception as err:
            print(f"Failed content image {filename}: {err}")

    style_rows = []
    for filename, artist_slug, url in STYLE_IMAGES:
        dst = style_dir / filename
        try:
            _download(url, dst)
        except Exception as err:
            print(f"Failed style image {filename}: {err}")
        if dst.exists():
            style_rows.append({"style_path": str(dst), "artist_slug": artist_slug})

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["style_path", "artist_slug"])
        writer.writeheader()
        writer.writerows(style_rows)
    print(f"Wrote style manifest: {manifest_path}")


if __name__ == "__main__":
    main()
