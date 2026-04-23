"""
Convert theseus_ocr_dataset to a HuggingFace-compatible dataset.

Output layout (HF_DATASET_FOLDER):
  train-00000-of-00001.parquet        — records with image paths
  images/                             — PNG image files
  README.md                           — dataset card with YAML metadata
"""

import json
import os
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

DATASET_JSON = "theseus_ocr_dataset/theseus_ocr_dataset.json"
CROPS_ROOT = "theseus_ocr_dataset"
HF_DATASET_FOLDER = "theseus_ocr_hf"
IMAGES_DIR = "images"


def save_image_file(rec: dict, out_dir: str) -> str | None:
    abs_src = os.path.join(CROPS_ROOT, rec["image"])
    images_dir = os.path.join(out_dir, IMAGES_DIR)
    os.makedirs(images_dir, exist_ok=True)

    # Parse pdf name, page, para index from the source image path
    # e.g. "paragraph_crops/theseus_474/p1_para1.png" → theseus_474, 1, 1
    src_path = rec["image"]
    parts = os.path.basename(src_path).replace(".png", "").split("_")
    # parts like ["p1", "para1"]
    page_part = parts[0]          # "p1"
    para_part = parts[1]          # "para1"
    page_num = page_part[1:]      # "1"
    para_num = para_part[4:]      # "1" (drop "para")
    pdf_name = os.path.splitext(rec["pdf_file"])[0]  # strip .pdf
    new_name = f"{pdf_name}_{page_num}_para{para_num.zfill(2)}.png"
    abs_dst = os.path.join(images_dir, new_name)

    if os.path.exists(abs_dst):
        return os.path.join(IMAGES_DIR, new_name)

    try:
        with Image.open(abs_src) as img:
            img.save(abs_dst, format="PNG")
        return os.path.join(IMAGES_DIR, new_name)
    except Exception as e:
        print(f"⚠️  Could not load {abs_src}: {e}")
        return None


MAX_SHARD_BYTES = 50 * 1024 * 1024  # 50 MB per shard


def _make_table(pdf_files, pages, texts, image_paths):
    return pa.table({
        "pdf_file":   pa.array(pdf_files,   type=pa.string()),
        "page":       pa.array(pages,       type=pa.int32()),
        "text":       pa.array(texts,       type=pa.string()),
        "image_path": pa.array(image_paths,  type=pa.string()),
    })


def write_parquet(records: list[dict], out_dir: str) -> int:
    shards: list[pa.Table] = []
    pdf_files, pages, texts, image_paths = [], [], [], []
    shard_bytes = 0

    def flush():
        if pdf_files:
            shards.append(_make_table(pdf_files, pages, texts, image_paths))
            pdf_files.clear()
            pages.clear()
            texts.clear()
            image_paths.clear()

    for rec in tqdm(records, desc="Saving images"):
        img_path = save_image_file(rec, out_dir)
        if img_path is None:
            continue
        img_size = os.path.getsize(os.path.join(out_dir, img_path))
        if shard_bytes + img_size > MAX_SHARD_BYTES and pdf_files:
            flush()
            shard_bytes = 0
        pdf_files.append(rec["pdf_file"])
        pages.append(rec["page"])
        texts.append(rec["text"])
        image_paths.append(img_path)
        shard_bytes += img_size

    flush()

    total_shards = len(shards)
    total_rows = 0
    for idx, table in enumerate(shards):
        path = os.path.join(out_dir, f"train-{idx + 1:05d}-of-{total_shards:05d}.parquet")
        pq.write_table(table, path, compression="snappy")
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"✅ Shard {idx + 1}/{total_shards}: {len(table)} rows, {size_mb:.1f} MB → {path}")
        total_rows += len(table)

    return total_rows


README_TEMPLATE = """\
---
license: cc-by-4.0
language:
- fi
task_categories:
- image-to-text
pretty_name: Theseus Finnish OCR
dataset_info:
  features:
    - name: pdf_file
      dtype: string
    - name: page
      dtype: int32
    - name: text
      dtype: string
    - name: image_path
      dtype: string
  splits:
    - name: train
      num_examples: {num_examples}
---

# Theseus Finnish OCR Dataset

Paragraph-level OCR dataset harvested from [Theseus.fi](https://www.theseus.fi),
the Finnish repository of university of applied sciences theses.

Each record is one paragraph crop extracted from a thesis PDF, paired with the
text extracted by `pdfplumber`.

## Image Resolution

Paragraph crops are rendered at **{resolution} DPI** (dots per inch) with 2 px
padding on each side. At {resolution} DPI a standard A4 page is
{page_w} × {page_h} pixels, giving high enough resolution for training
OCR and document-understanding models.

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `pdf_file` | string | Source PDF filename |
| `page` | int | Page number (1-indexed) |
| `image_path` | string | Path to paragraph crop PNG under `images/` at {resolution} DPI with 2 px padding on each side |
| `text` | string | Extracted paragraph text from `pdfplumber` |

## Source

Records were harvested via the Theseus OAI-PMH endpoint and PDFs were
downloaded from the DSpace bitstream API.
"""


def write_readme(out_dir: str, num_examples: int, resolution: int = 300) -> None:
    # A4 at given DPI (8.27 × 11.69 inches)
    page_w = round(8.27 * resolution)
    page_h = round(11.69 * resolution)
    path = os.path.join(out_dir, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(README_TEMPLATE.format(
            num_examples=num_examples,
            resolution=resolution,
            page_w=page_w,
            page_h=page_h,
        ))
    print(f"✅ Wrote {path}")


def main() -> None:
    with open(DATASET_JSON, encoding="utf-8") as f:
        records = json.load(f)

    print(f"Converting {len(records)} records → '{HF_DATASET_FOLDER}/'")
    write_parquet(records, HF_DATASET_FOLDER)
    write_readme(HF_DATASET_FOLDER, num_examples=len(records))
    print(f"\nDone. Push to HuggingFace Hub with:")
    print(f"  huggingface-cli upload <your-username>/theseus-finnish-ocr {HF_DATASET_FOLDER} --repo-type dataset")


if __name__ == "__main__":
    main()
