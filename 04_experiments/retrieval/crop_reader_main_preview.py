"""Remove spreadsheet row/column headers from the publication-table PNG preview."""

from pathlib import Path

from PIL import Image


ROOT = Path(r"D:\memorytable")
OUTPUT = ROOT / "outputs" / "019fb6b6-9b0d-7412-b41f-5e231a85061b"
SOURCE = OUTPUT / "Reader_Main.png"


def main() -> None:
    image = Image.open(SOURCE)
    # artifact-tool renders 40 px of row labels and 20 px of column labels.
    cropped = image.crop((40, 20, image.width, image.height))
    cropped.save(SOURCE)
    print(f"Cropped {image.size} -> {cropped.size}: {SOURCE}")


if __name__ == "__main__":
    main()
