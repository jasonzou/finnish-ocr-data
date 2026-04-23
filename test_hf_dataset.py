"""Quick smoke-test for the local HuggingFace dataset."""

from datasets import load_dataset, features

HF_DATASET_FOLDER = "theseus_ocr_hf"


def main() -> None:
    print(f"Loading dataset from '{HF_DATASET_FOLDER}/'...")
    ds = load_dataset("parquet", data_files=f"{HF_DATASET_FOLDER}/data/*.parquet", split="train")
    ds = ds.cast_column("image", features.Image())

    print(f"  Rows      : {len(ds)}")
    print(f"  Features  : {ds.features}")

    print("\nSample records:")
    for i, row in enumerate(ds.select(range(6))):
        img = row["image"]
        out_path = f"test_sample_{i}.jpg"
        img.convert("RGB").save(out_path, format="JPEG")
        print(f"\n  [{i}] pdf_file : {row['pdf_file']}")
        print(f"       page     : {row['page']}")
        print(f"       text     : {row['text'][:80]!r}")
        print(f"       image    : {img.size} px, mode={img.mode} → saved to {out_path}")


if __name__ == "__main__":
    main()
