"""
Convert theseus_ocr_dataset to a HuggingFace-compatible dataset.

Output layout (HF_DATASET_FOLDER):
  data/train-00000-of-00001.parquet   — records with embedded image bytes
  README.md                           — dataset card with YAML metadata
"""

import io
import json
import os
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

DATASET_JSON = "theseus_ocr_dataset/theseus_ocr_dataset.json"
CROPS_ROOT = "theseus_ocr_dataset"
HF_DATASET_FOLDER = "theseus_ocr_hf"


def load_image_bytes(image_rel_path: str) -> bytes | None:
    abs_path = os.path.join(CROPS_ROOT, image_rel_path)
    try:
        with Image.open(abs_path) as img:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
    except Exception as e:
        print(f"⚠️  Could not load {abs_path}: {e}")
        return None


def write_parquet(records: list[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    pdf_files, pages, texts, images = [], [], [], []
    for rec in tqdm(records, desc="Loading images"):
        img_bytes = load_image_bytes(rec["image"])
        if img_bytes is None:
            continue
        pdf_files.append(rec["pdf_file"])
        pages.append(rec["page"])
        texts.append(rec["text"])
        # HF Image feature is stored as a struct {bytes, path}
        images.append({"bytes": img_bytes, "path": os.path.basename(rec["image"])})

    table = pa.table({
        "pdf_file": pa.array(pdf_files, type=pa.string()),
        "page":     pa.array(pages,     type=pa.int32()),
        "text":     pa.array(texts,     type=pa.string()),
        "image":    pa.array(images,    type=pa.struct([
                        pa.field("bytes", pa.binary()),
                        pa.field("path",  pa.string()),
                    ])),
    })

    parquet_path = os.path.join(out_dir, "data", "train-00000-of-00001.parquet")
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    pq.write_table(table, parquet_path, compression="snappy")
    print(f"✅ Wrote {len(table)} rows → {parquet_path}")


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
    - name: image
      dtype: image
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
| `image` | Image | Paragraph crop at {resolution} DPI with 2 px padding on each side |
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
