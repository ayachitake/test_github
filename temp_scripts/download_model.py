import kagglehub
import shutil
from pathlib import Path

target_dir = Path("d:/03_Development/translate_akkadian/input/datasets")
target_dir.mkdir(parents=True, exist_ok=True)

print("Downloading assiaben/final-byt5 dataset...")
path = kagglehub.dataset_download("assiaben/final-byt5", force_download=True)

print(f"Downloaded to: {path}")

if path != target_dir:
    print(f"Moving files to {target_dir}...")
    for item in Path(path).iterdir():
        dest = target_dir / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(item), str(dest))
    print(f"Files moved successfully to {target_dir}")

print(f"\nFinal location: {target_dir}")
print(f"Contents:")
for item in target_dir.iterdir():
    print(f"  - {item.name}")
