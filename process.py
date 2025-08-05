import os
import re
import shutil
from pathlib import Path

def clean_folder_names(dataset_dir):
    def clean_name(name):
        return re.sub(r"[^a-zA-Z0-9_]", "", name)

    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        if os.path.isdir(folder_path):
            cleaned = clean_name(folder)
            new_path = os.path.join(dataset_dir, cleaned)
            if cleaned != folder:
                print(f"Renaming: {folder} -> {cleaned}")
                os.rename(folder_path, new_path)

def rename_images_by_class(train_dir):
    train_dir = Path(train_dir)

    for class_dir in train_dir.iterdir():
        if class_dir.is_dir():
            image_files = sorted([
                f for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']
            ])
            
            for idx, img_path in enumerate(image_files, 1):
                new_name = f"{class_dir.name}_{idx:04d}{img_path.suffix.lower()}"
                new_path = class_dir / new_name
                if img_path != new_path:
                    img_path.rename(new_path)
            print(f"Renamed {len(image_files)} images in '{class_dir.name}'")

def sync_folders_by_reference(main_dir, reference_dir):
    main_subfolders = set(os.listdir(main_dir))
    ref_subfolders = set(os.listdir(reference_dir))
    
    to_delete = main_subfolders - ref_subfolders
    
    for folder in to_delete:
        print(f"Removing: {folder}")
        full_path = os.path.join(main_dir, folder)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)

if __name__ == "__main__":

    clean_folder_names('./test')
    rename_images_by_class('./train')
    sync_folders_by_reference('./train', './test')
    
    print("All operations completed")