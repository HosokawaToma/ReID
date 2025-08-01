from dataclasses import dataclass
from pathlib import Path
import shutil

@dataclass
class FolderToDatasetConfig:
    class Directories:
        input_dir_str: str = "resources/dataset_folder/input"
        output_dir_str: str = "resources/dataset_folder/output"


class FolderToDataset:
    def __init__(self):
        self.input_dir_str = FolderToDatasetConfig.Directories.input_dir_str
        self.output_dir_str = FolderToDatasetConfig.Directories.output_dir_str

    def run(self):
        self.input_path = Path(self.input_dir_str)
        self.output_path = Path(self.output_dir_str)

        for i, person_dir in enumerate(self.input_path.iterdir()):
            person_id = i + 1
            if not person_dir.is_dir():
                continue

            for image_file in person_dir.glob("*.jpg"):
                if not image_file.is_file():
                    continue

                output_dir = self.output_path / f"{person_id}"
                if not output_dir.exists():
                    output_dir.mkdir(parents=True)

                output_file_name = f"{person_id}_{image_file.name}"
                output_file_path = output_dir / output_file_name

                if output_file_path.exists():
                    continue

                shutil.copy(image_file, output_file_path)

if __name__ == "__main__":
    folder_to_dataset = FolderToDataset()
    folder_to_dataset.run()
