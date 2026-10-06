#!/usr/bin/env python3
"""ดาวน์โหลดชุดข้อมูลภาพโรคพืชสำหรับเทรนโมเดลจำแนกภาพ

สคริปต์นี้ไม่ได้จำเป็นต่อการใช้งานระบบ (ระบบใช้โมเดล Claude วิเคราะห์ภาพได้ทันที)
แต่มีไว้สำหรับผู้ที่ต้องการเทรนโมเดลจำแนกภาพของตนเองเพื่อใช้งานแบบออฟไลน์
หรือใช้ร่วมกับผลจาก AI (ดู scripts/train_model.py)

ใช้งาน:
    python3 scripts/fetch_data.py list                 # ดูรายการแหล่งข้อมูลที่รองรับ
    python3 scripts/fetch_data.py verify               # ตรวจว่าเข้าถึงแหล่งข้อมูลได้หรือไม่
    python3 scripts/fetch_data.py fetch plantdoc       # ดาวน์โหลดชุดข้อมูลตามรหัส
    python3 scripts/fetch_data.py fetch-kaggle <slug>  # ดาวน์โหลดจาก Kaggle ด้วย slug ที่ระบุ
    python3 scripts/fetch_data.py fetch-url <url>      # ดาวน์โหลดไฟล์ zip/tar จาก URL โดยตรง
    python3 scripts/fetch_data.py search               # คำค้นแนะนำสำหรับหาชุดข้อมูลแตงโม

หมายเหตุสำคัญเรื่องเครือข่าย: สภาพแวดล้อมบางแห่ง (เช่น เครื่องที่รันในคลาวด์ที่จำกัด egress)
อาจบล็อก Kaggle, Hugging Face, Mendeley หรือ Zenodo สคริปต์จะรายงานให้ทราบอย่างชัดเจน
ให้ดาวน์โหลดบนเครื่องที่เข้าถึงได้แล้วคัดลอกมาไว้ที่ datasets/ แทน
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import ssl
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets"
USER_AGENT = "melon-disease-ai/1.0 (dataset fetcher)"

SOURCES: dict[str, dict[str, Any]] = {
    "plantdoc": {
        "name": "PlantDoc Dataset",
        "kind": "git",
        "repo": "https://github.com/pratikkayal/PlantDoc-Dataset",
        "url": "https://codeload.github.com/pratikkayal/PlantDoc-Dataset/zip/refs/heads/master",
        "about": (
            "ภาพถ่ายโรคพืชในสภาพแปลงจริง 27 คลาส 13 ชนิดพืช เหมาะกับการเทรนให้ทนต่อภาพถ่ายจริง "
            "ไม่มีคลาสแตงโมโดยตรง แต่ใช้เป็นชุด pretrain ที่ใกล้เคียงสภาพจริงได้ดี"
        ),
        "license": "CC BY 4.0 (ตรวจสอบเงื่อนไขในรีโพก่อนใช้เชิงพาณิชย์)",
        "size_hint": "ประมาณ 600 MB",
    },
    "plantvillage-github": {
        "name": "PlantVillage Dataset (GitHub mirror)",
        "kind": "git",
        "repo": "https://github.com/spMohanty/PlantVillage-Dataset",
        "url": "https://codeload.github.com/spMohanty/PlantVillage-Dataset/zip/refs/heads/master",
        "about": (
            "ภาพใบพืชในห้องปฏิบัติการ 54,000+ ภาพ 38 คลาส เป็นชุดมาตรฐานสำหรับ pretrain "
            "ข้อจำกัด: ถ่ายบนพื้นหลังเรียบ ทำให้โมเดลที่เทรนจากชุดนี้อย่างเดียวมักไม่แม่นกับภาพถ่ายในแปลงจริง"
        ),
        "license": "CC BY-SA 3.0",
        "size_hint": "ประมาณ 2 GB",
    },
    "plantvillage-kaggle": {
        "name": "PlantVillage Dataset (Kaggle)",
        "kind": "kaggle",
        "slug": "abdallahalidev/plantvillage-dataset",
        "about": "ชุดเดียวกับด้านบนแต่จัดระเบียบโฟลเดอร์มาแล้ว ดาวน์โหลดเร็วกว่าผ่าน Kaggle API",
        "license": "CC BY-SA 3.0",
        "size_hint": "ประมาณ 2 GB",
    },
}

SEARCH_HINTS = [
    ("Kaggle", "https://www.kaggle.com/datasets?search=watermelon+leaf+disease"),
    ("Kaggle", "https://www.kaggle.com/datasets?search=cucurbit+disease"),
    ("Roboflow Universe", "https://universe.roboflow.com/search?q=watermelon+disease"),
    ("Mendeley Data", "https://data.mendeley.com/research-data/?search=watermelon%20leaf%20disease"),
    ("Hugging Face", "https://huggingface.co/datasets?search=plant%20disease"),
    ("Zenodo", "https://zenodo.org/search?q=watermelon+disease+images"),
]

CLASS_MAPPING_HINT = """
เมื่อดาวน์โหลดเสร็จแล้ว ให้จัดโฟลเดอร์ภาพให้เป็นรูปแบบนี้ก่อนเทรน
โดยตั้งชื่อโฟลเดอร์ให้ตรงกับรหัสโรค (id) ในคลังความรู้ เช่น

    datasets/watermelon/train/anthracnose/*.jpg
    datasets/watermelon/train/downy_mildew/*.jpg
    datasets/watermelon/train/healthy/*.jpg
    datasets/watermelon/val/anthracnose/*.jpg
    ...

ดูรหัสโรคทั้งหมดได้จาก:  python3 scripts/train_model.py --list-classes
"""


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _opener() -> urllib.request.OpenerDirector:
    """สร้าง opener ที่เคารพ proxy และ CA bundle ของระบบ"""
    ca_file = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    context = ssl.create_default_context(cafile=ca_file) if ca_file else ssl.create_default_context()
    handlers: list[urllib.request.BaseHandler] = [urllib.request.HTTPSHandler(context=context)]
    proxies = {}
    for key in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
        value = os.environ.get(key)
        if value:
            proxies[key.split("_")[0].lower()] = value
    if proxies:
        handlers.append(urllib.request.ProxyHandler(proxies))
    opener = urllib.request.build_opener(*handlers)
    opener.addheaders = [("User-Agent", USER_AGENT)]
    return opener


def download(url: str, dest: Path, timeout: int = 60) -> Path:
    """ดาวน์โหลดไฟล์พร้อมแสดงความคืบหน้า"""
    ensure_dir(dest.parent)
    opener = _opener()
    print(f"==> ดาวน์โหลด {url}")
    started = time.time()
    try:
        with opener.open(url, timeout=timeout) as response, dest.open("wb") as fh:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    print(f"\r    {done/1e6:.1f}/{total/1e6:.1f} MB ({pct:.1f}%)", end="", flush=True)
                else:
                    print(f"\r    {done/1e6:.1f} MB", end="", flush=True)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"ดาวน์โหลดไม่สำเร็จ: HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"เชื่อมต่อไม่สำเร็จ: {exc.reason}\n"
            "สาเหตุที่พบบ่อยคือนโยบายเครือข่ายของเครื่องนี้บล็อกปลายทางอยู่ "
            "ให้ดาวน์โหลดจากเครื่องอื่นแล้วคัดลอกไฟล์มาไว้ที่โฟลเดอร์ datasets/"
        ) from exc
    print(f"\n    เสร็จใน {time.time() - started:.1f} วินาที -> {dest}")
    return dest


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, target: Path) -> None:
    """แตกไฟล์โดยป้องกัน path traversal (zip slip)"""
    ensure_dir(target)
    resolved_target = target.resolve()
    print(f"==> แตกไฟล์ไปที่ {target}")
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                out = (target / member).resolve()
                if not str(out).startswith(str(resolved_target)):
                    raise SystemExit(f"ไฟล์บีบอัดมีเส้นทางที่ไม่ปลอดภัย: {member}")
            zf.extractall(target)
    elif archive.name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                out = (target / member.name).resolve()
                if not str(out).startswith(str(resolved_target)):
                    raise SystemExit(f"ไฟล์บีบอัดมีเส้นทางที่ไม่ปลอดภัย: {member.name}")
            tf.extractall(target)
    else:
        print("    ไม่ใช่ไฟล์บีบอัดที่รองรับ ข้ามขั้นตอนการแตกไฟล์")


def write_manifest(dataset_dir: Path, info: dict[str, Any]) -> None:
    manifest = dataset_dir / "manifest.json"
    info["fetched_at"] = datetime.now(timezone.utc).isoformat()
    images = sum(
        1
        for p in dataset_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )
    info["image_count"] = images
    manifest.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"==> บันทึก manifest: {manifest} (พบภาพ {images} ไฟล์)")


def fetch_github(source_id: str, source: dict[str, Any], keep_archive: bool) -> None:
    target = ensure_dir(DATASET_DIR / source_id)
    archive = target / "download.zip"
    download(source["url"], archive)
    checksum = sha256_of(archive)
    safe_extract(archive, target)
    if not keep_archive:
        archive.unlink(missing_ok=True)
    write_manifest(
        target,
        {
            "source_id": source_id,
            "name": source["name"],
            "url": source["url"],
            "license": source.get("license", ""),
            "sha256": checksum,
        },
    )


def fetch_git(source_id: str, repo: str, depth: int = 1, branch: str | None = None) -> None:
    """ดึงชุดข้อมูลด้วย git clone แบบตื้น

    วิธีนี้ใช้ได้ในหลายเครือข่ายที่บล็อกการดาวน์โหลดไฟล์ zip จาก GitHub โดยตรง
    เพราะใช้โปรโตคอล git ผ่าน HTTPS ซึ่งมักได้รับอนุญาต
    """
    if shutil.which("git") is None:
        raise SystemExit("ไม่พบคำสั่ง git บนเครื่องนี้ กรุณาติดตั้ง git ก่อน")
    target = DATASET_DIR / source_id
    if target.exists() and any(target.iterdir()):
        print(f"==> มีข้อมูลอยู่แล้วที่ {target} (ข้ามการดาวน์โหลด ใช้ --force เพื่อดาวน์โหลดใหม่)")
        return
    ensure_dir(DATASET_DIR)
    cmd = ["git", "clone", "--depth", str(depth)]
    if branch:
        cmd += ["--branch", branch]
    cmd += [repo, str(target)]
    print(f"==> รัน: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(
            "git clone ไม่สำเร็จ\n"
            "ลองใช้คำสั่ง fetch-url กับไฟล์ zip แทน หรือดาวน์โหลดจากเครื่องอื่นแล้วคัดลอกมาที่ datasets/"
        )
    write_manifest(target, {"source_id": source_id, "name": source_id, "url": repo, "method": "git"})


def fetch_kaggle(source_id: str, slug: str) -> None:
    if shutil.which("kaggle") is None:
        raise SystemExit(
            "ไม่พบคำสั่ง kaggle\n"
            "ติดตั้งด้วย:  pip install kaggle\n"
            "แล้วตั้งค่าคีย์ที่ ~/.kaggle/kaggle.json หรือตัวแปร KAGGLE_USERNAME และ KAGGLE_KEY\n"
            "(ขอคีย์ได้ที่ https://www.kaggle.com/settings/account)"
        )
    target = ensure_dir(DATASET_DIR / source_id)
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(target), "--unzip"]
    print(f"==> รัน: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(
            "ดาวน์โหลดจาก Kaggle ไม่สำเร็จ\n"
            "ตรวจสอบว่า: ตั้งค่าคีย์ถูกต้อง, ยอมรับเงื่อนไขของชุดข้อมูลบนเว็บแล้ว, "
            "และเครือข่ายของเครื่องนี้เข้าถึง kaggle.com ได้"
        )
    write_manifest(target, {"source_id": source_id, "name": slug, "url": f"kaggle:{slug}"})


def cmd_list() -> int:
    print("แหล่งข้อมูลที่รองรับ:\n")
    for key, source in SOURCES.items():
        print(f"  {key}")
        print(f"      ชื่อ      : {source['name']}")
        print(f"      ประเภท    : {source['kind']}")
        print(f"      ขนาดโดยประมาณ: {source.get('size_hint','ไม่ทราบ')}")
        print(f"      สัญญาอนุญาต: {source.get('license','ตรวจสอบที่ต้นทาง')}")
        print(f"      รายละเอียด : {source['about']}")
        print()
    print(CLASS_MAPPING_HINT)
    return 0


def cmd_search() -> int:
    print("ยังไม่มีชุดข้อมูลภาพโรคแตงโมมาตรฐานที่เปิดให้ดาวน์โหลดได้โดยตรงแบบเสถียร")
    print("แนะนำให้ค้นหาจากแหล่งต่อไปนี้ และตรวจสอบสัญญาอนุญาตก่อนนำไปใช้:\n")
    for name, url in SEARCH_HINTS:
        print(f"  {name:20s} {url}")
    print(
        "\nทางเลือกที่ได้ผลดีที่สุดในทางปฏิบัติคือถ่ายภาพจากแปลงของตนเอง "
        "โดยเก็บอย่างน้อย 100-200 ภาพต่อโรค ถ่ายหลายมุม หลายช่วงเวลา และหลายระดับความรุนแรง"
    )
    print(CLASS_MAPPING_HINT)
    return 0


def cmd_verify() -> int:
    opener = _opener()
    hosts = {
        "GitHub (codeload)": "https://codeload.github.com",
        "Kaggle": "https://www.kaggle.com",
        "Hugging Face": "https://huggingface.co",
        "Zenodo": "https://zenodo.org",
        "Mendeley Data": "https://data.mendeley.com",
    }
    print("ตรวจการเข้าถึงแหล่งข้อมูล (ใช้เวลาสักครู่)\n")
    blocked = []
    for name, url in hosts.items():
        try:
            request = urllib.request.Request(url, method="HEAD")
            with opener.open(request, timeout=12) as response:
                print(f"  [เข้าถึงได้]   {name:20s} HTTP {response.status}")
        except urllib.error.HTTPError as exc:
            print(f"  [เข้าถึงได้]   {name:20s} HTTP {exc.code}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [เข้าถึงไม่ได้] {name:20s} {type(exc).__name__}: {exc}")
            blocked.append(name)
    git_ok = shutil.which("git") is not None
    print(f"\n  [{'พร้อม' if git_ok else 'ไม่พบ'}]        git (ใช้สำหรับ fetch-git ซึ่งมักผ่านนโยบายเครือข่ายได้ดีกว่าการโหลด zip)")
    if blocked:
        print(
            "\nแหล่งที่เข้าถึงไม่ได้: "
            + ", ".join(blocked)
            + "\nหากเป็นข้อจำกัดของนโยบายเครือข่าย ให้ดาวน์โหลดจากเครื่องอื่นแล้วคัดลอกมาไว้ที่ datasets/"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ดาวน์โหลดชุดข้อมูลภาพโรคพืชสำหรับเทรนโมเดล",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="แสดงรายการแหล่งข้อมูลที่รองรับ")
    sub.add_parser("search", help="คำค้นแนะนำสำหรับหาชุดข้อมูลแตงโม")
    sub.add_parser("verify", help="ตรวจว่าเครื่องนี้เข้าถึงแหล่งข้อมูลใดได้บ้าง")

    p_fetch = sub.add_parser("fetch", help="ดาวน์โหลดตามรหัสแหล่งข้อมูล")
    p_fetch.add_argument("source_id", choices=sorted(SOURCES))
    p_fetch.add_argument("--keep-archive", action="store_true", help="เก็บไฟล์บีบอัดไว้หลังแตกไฟล์")
    p_fetch.add_argument("--force", action="store_true", help="ดาวน์โหลดใหม่แม้มีข้อมูลเดิมอยู่แล้ว")

    p_kaggle = sub.add_parser("fetch-kaggle", help="ดาวน์โหลดจาก Kaggle ด้วย slug")
    p_kaggle.add_argument("slug", help="เช่น abdallahalidev/plantvillage-dataset")

    p_git = sub.add_parser("fetch-git", help="ดึงชุดข้อมูลจาก git repository")
    p_git.add_argument("repo", help="เช่น https://github.com/user/dataset-repo")
    p_git.add_argument("--name", default=None, help="ชื่อโฟลเดอร์ปลายทางใน datasets/")
    p_git.add_argument("--branch", default=None, help="สาขาที่ต้องการ")
    p_git.add_argument("--depth", type=int, default=1, help="จำนวน commit ที่ดึง (ค่าเริ่มต้น 1)")

    p_url = sub.add_parser("fetch-url", help="ดาวน์โหลดไฟล์บีบอัดจาก URL")
    p_url.add_argument("url")
    p_url.add_argument("--name", default="custom", help="ชื่อโฟลเดอร์ปลายทางใน datasets/")

    args = parser.parse_args()

    if args.command == "list":
        return cmd_list()
    if args.command == "search":
        return cmd_search()
    if args.command == "verify":
        return cmd_verify()
    if args.command == "fetch":
        source = SOURCES[args.source_id]
        if source["kind"] == "kaggle":
            fetch_kaggle(args.source_id, source["slug"])
        elif source["kind"] == "git":
            if args.force:
                shutil.rmtree(DATASET_DIR / args.source_id, ignore_errors=True)
            try:
                fetch_git(args.source_id, source["repo"])
            except SystemExit as exc:
                print(f"{exc}\n==> ลองวิธีสำรองด้วยการดาวน์โหลดไฟล์ zip")
                fetch_github(args.source_id, source, args.keep_archive)
        else:
            fetch_github(args.source_id, source, args.keep_archive)
        return 0
    if args.command == "fetch-git":
        name = args.name or args.repo.rstrip("/").split("/")[-1].replace(".git", "")
        fetch_git(name, args.repo, depth=args.depth, branch=args.branch)
        return 0
    if args.command == "fetch-kaggle":
        fetch_kaggle(args.slug.replace("/", "_"), args.slug)
        return 0
    if args.command == "fetch-url":
        target = ensure_dir(DATASET_DIR / args.name)
        archive = target / Path(args.url).name.split("?")[0]
        download(args.url, archive)
        checksum = sha256_of(archive)
        safe_extract(archive, target)
        write_manifest(target, {"source_id": args.name, "url": args.url, "sha256": checksum})
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
