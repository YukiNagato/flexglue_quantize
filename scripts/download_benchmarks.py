"""Download and audit the complete pair-level benchmarks without installing packages.

HardMatch's final pair manifest always includes both original splits (1000 pairs).
Upstream archives and annotations are kept unchanged.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import stat
import subprocess
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "megadepth1500": "https://cvg-data.inf.ethz.ch/megadepth/megadepth1500.zip",
    "scannet1500": "https://cvg-data.inf.ethz.ch/scannet/scannet1500.zip",
    "hardmatch": "https://github.com/davnords/storage/releases/download/hardmatch/hardmatch.zip",
}
EXPECTED = {"megadepth1500": 1500, "scannet1500": 1500, "hardmatch": 1000}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def download(name, root):
    folder = root / ".downloads"
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / f"{name}.zip"
    # A complete ZIP is re-used; an interrupted download can be resumed.
    if not zipfile.is_zipfile(archive):
        print("DOWNLOAD", name, flush=True)
        with archive.with_suffix(".log").open("a") as log:
            subprocess.run([
                "curl", "-4", "-L", "--fail", "--show-error", "--retry", "5",
                "--retry-delay", "3", "--connect-timeout", "20", "--speed-limit", "1024",
                "--speed-time", "60", "--continue-at", "-", "--output", str(archive), SOURCES[name],
            ], stdout=log, stderr=log, check=True)
    return archive


def extract(name, archive, root):
    destination = root / name
    marker = destination / ".archive.json"
    checksum = digest(archive)
    if marker.exists():
        if json.loads(marker.read_text())["sha256"] != checksum:
            raise ValueError(f"Archive changed for existing dataset: {destination}")
        return destination, checksum
    if destination.exists():
        raise FileExistsError(f"Refusing to replace an unverified directory: {destination}")
    staging = root / ".extracting" / name
    staging.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            relative = PurePosixPath(member.filename)
            if (relative.is_absolute() or ".." in relative.parts
                    or not relative.parts or relative.parts[0] != name
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError(f"Unexpected archive entry: {member.filename}")
        # Reading every entry checks its ZIP CRC, including non-image assets.
        zipped.extractall(staging)
    (staging / name).rename(destination)
    write_json(marker, dict(url=SOURCES[name], sha256=checksum, bytes=archive.stat().st_size,
                            zip_crc_verified=True))
    return destination, checksum


def calibrated_pairs(folder, name):
    rows = []
    for line in (folder / "pairs_calibrated.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        columns = line.split()
        if len(columns) not in (32, 36):
            raise ValueError(f"Unexpected calibrated pair format: {name}, {len(columns)} columns")
        values = np.asarray(columns[2:], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite camera calibration")
        images = []
        for relative in columns[:2]:
            candidates = [folder / relative, folder / "images" / relative]
            found = next((p for p in candidates if p.is_file()), None)
            if found is None:
                raise FileNotFoundError(relative)
            images.append(str(found.relative_to(folder)))
        if len(columns) == 36:
            transform = values[18:].reshape(4, 4)
        else:
            transform = np.eye(4)
            transform[:3, :3] = values[18:27].reshape(3, 3)
            transform[:3, 3] = values[27:30]
        rows.append(dict(pair_id=f"{name}_{len(rows):04d}", image0=images[0], image1=images[1],
                         K0=values[:9].reshape(3, 3).tolist(), K1=values[9:18].reshape(3, 3).tolist(),
                         T_0to1=transform.tolist(), source_split="test"))
    return rows


def hardmatch_assets(directory):
    files = [directory / x for x in ("A_lr.jpg", "B_lr.jpg", "corresp_lr.txt")]
    if all(p.is_file() for p in files):
        return files, "official_archive"
    # The current release omits the images for one annotated pair. Recover the
    # exact sources named by its metadata and keep all derived assets separate.
    recovered = []
    scales = []
    provenance = []
    for letter in "AB":
        metadata = json.loads((directory / f"{letter}_metadata.json").read_text())
        url = metadata["original_url"]
        if not url.startswith("https://upload.wikimedia.org/"):
            raise ValueError(f"Unexpected recovery source: {url}")
        source = directory / f"{letter}_source.jpg"
        if not source.exists():
            temporary = source.with_suffix(".jpg.part")
            subprocess.run([
                "curl", "-4", "-L", "--fail", "--show-error", "--retry", "3",
                "--connect-timeout", "20", "--max-time", "120",
                "--output", str(temporary), url,
            ], check=True)
            with Image.open(temporary) as image:
                image.load()
            temporary.replace(source)
        target = directory / f"{letter}_recovered_lr.jpg"
        with Image.open(source) as image:
            original_size = image.size
            size = tuple(round(v * 1024 / max(image.size)) for v in image.size)
            image.convert("RGB").resize(size, Image.Resampling.LANCZOS).save(target, quality=95)
        scales.extend([size[0] / original_size[0], size[1] / original_size[1]])
        recovered.append(target)
        provenance.append(dict(url=url, source_sha256=digest(source),
                               original_size=original_size, resized_size=size,
                               output=target.name, output_sha256=digest(target)))
    text = (directory / "corresp.txt").read_text().strip()
    points = np.fromstring(text, sep=" ", dtype=np.float64).reshape(-1, 4)
    target = directory / "corresp_recovered_lr.txt"
    np.savetxt(target, points * np.asarray(scales), fmt="%.8f")
    recovered.append(target)
    write_json(directory / "recovery.json", dict(
        reason="Official archive contains annotations and metadata but omits low-resolution assets.",
        images=provenance, coordinate_scales=scales,
        transform="PIL RGB, longest side 1024, rounded dimensions, LANCZOS, JPEG quality 95",
        note="Locally reconstructed assets; not byte-identical official benchmark assets."))
    return recovered, "recovered_from_original_sources"


def hardmatch_pairs(folder):
    # This reproduces upstream's category-based split while retaining all pairs.
    directories = sorted(p.parent for p in folder.rglob("corresp.txt"))
    groups = sorted({p.parent.parent for p in directories})
    if len(groups) != 100:
        raise ValueError(f"Expected 100 HardMatch groups, found {len(groups)}")
    shuffled = groups.copy()
    random.Random(1337).shuffle(shuffled)
    validation = set(shuffled[:10])
    rows = []
    for directory in directories:
        files, asset_source = hardmatch_assets(directory)
        for file in files:
            if not file.is_file():
                raise FileNotFoundError(file)
        text = files[2].read_text().strip()
        points = np.fromstring(text, sep=" ", dtype=np.float64).reshape(-1, 4)
        if not np.isfinite(points).all():
            raise ValueError(f"Invalid correspondence annotation: {files[2]}")
        category_file = directory / "categories.txt"
        categories = [s.strip() for s in category_file.read_text().splitlines() if s.strip()] if category_file.exists() else []
        rows.append(dict(pair_id=str(directory.relative_to(folder)),
                         image0=str(files[0].relative_to(folder)), image1=str(files[1].relative_to(folder)),
                         correspondences=str(files[2].relative_to(folder)), gt_correspondences=len(points),
                         asset_source=asset_source,
                         categories=categories, source_split="val" if directory.parent.parent in validation else "test",
                         final_evaluation=True,
                         upstream_fundamental_metric_eligible=len(points) >= 8 and "dynamic" not in categories))
    return rows


def audit(folder, name, archive, checksum):
    rows = hardmatch_pairs(folder) if name == "hardmatch" else calibrated_pairs(folder, name)
    if len(rows) != EXPECTED[name]:
        raise ValueError(f"{name}: expected {EXPECTED[name]} pairs, found {len(rows)}")
    referenced = {r[key] for r in rows for key in ("image0", "image1")}
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))

    def check_image(path):
        with Image.open(path) as image:
            image.load()
            if min(image.size) <= 0:
                raise ValueError(f"Invalid image: {path}")
            size = image.size
        return str(path.relative_to(folder)), dict(width=size[0], height=size[1], sha256=digest(path), bytes=path.stat().st_size)

    with ThreadPoolExecutor(8) as pool:
        images = dict(pool.map(check_image, files))
    if missing := referenced - images.keys():
        raise ValueError(f"Missing images: {sorted(missing)}")
    depths = list(folder.rglob("*.h5"))
    # Verify the array payload, not just the ZIP container, where depth exists.
    if depths:
        try:
            import h5py
        except ImportError:
            import sys
            fallback = ROOT / "dataset_sources/python_deps"
            if not fallback.is_dir():
                raise ImportError("Depth validation requires h5py; no packages were installed automatically.")
            sys.path.insert(0, str(fallback))
            import h5py
        for path in depths:
            with h5py.File(path, "r") as handle:
                depth = handle["depth"][:]
                if depth.ndim != 2 or not np.isfinite(depth).all():
                    raise ValueError(f"Invalid depth: {path}")
    (folder / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_json(folder / "images.json", images)
    report = dict(state="complete", dataset=name, source_url=SOURCES[name], archive_sha256=checksum,
                  archive_bytes=archive.stat().st_size, pairs=len(rows),
                  source_splits=dict(Counter(r["source_split"] for r in rows)),
                  final_evaluation_split="all" if name == "hardmatch" else "test",
                  final_evaluation_pairs=len(rows), referenced_images=len(referenced),
                  decoded_images=len(images), decoded_depth_maps=len(depths), zip_crc_verified=True,
                  extracted_bytes=sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()),
                  pair_manifest_sha256=digest(folder / "pairs.jsonl"))
    if name == "hardmatch":
        report["upstream_fundamental_metric_eligible_pairs"] = sum(r["upstream_fundamental_metric_eligible"] for r in rows)
        report["all_pairs_retained"] = True
        report["asset_sources"] = dict(Counter(r["asset_source"] for r in rows))
        report["pairs_without_gt_correspondences"] = sum(r["gt_correspondences"] == 0 for r in rows)
        report["selection_note"] = "Use this all-pair manifest; upstream dataset loader defaults to 900 test pairs. No pairs removed here."
    write_json(folder / "manifest.json", report)
    print("READY", name, len(rows), "pairs", len(images), "images", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/benchmarks")
    parser.add_argument("--datasets", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument("--download-workers", type=int, default=3)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.download_workers) as pool:
        archives = list(pool.map(lambda name: download(name, args.root), args.datasets))
    reports = {
        name: json.loads(path.read_text())
        for name in SOURCES
        if (path := args.root / name / "manifest.json").is_file()
    }
    for name, archive in zip(args.datasets, archives):
        print("EXTRACT/AUDIT", name, flush=True)
        folder, checksum = extract(name, archive, args.root)
        reports[name] = audit(folder, name, archive, checksum)
        write_json(args.root / "download_report.json", reports)


if __name__ == "__main__":
    main()
