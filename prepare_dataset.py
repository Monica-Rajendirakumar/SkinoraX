import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
import random

# Configuration
METADATA_PATH = 'HAM10000_metadata.csv'
IMAGE_FOLDERS = ['HAM10000_images_part_1', 'HAM10000_images_part_2']
OUTPUT_DIR = 'skin_dataset'
SAMPLE_SIZE = 500  # Total images to use (250 benign + 250 malignant)
RANDOM_SEED = 42

# Class mapping
CLASS_MAPPING = {
    'mel': 'malignant',  # Melanoma
    'nv': 'benign'       # Nevus (benign mole)
}

def find_image_path(image_id, folders):
    """Find the full path of an image across multiple folders"""
    for folder in folders:
        path = os.path.join(folder, f"{image_id}.jpg")
        if os.path.exists(path):
            return path
    return None

def prepare_dataset():
    print("=" * 60)
    print("SKINORAX - DATASET PREPARATION")
    print("=" * 60)
    
    # Load metadata
    print("\n[1/6] Loading metadata...")
    df = pd.read_csv(METADATA_PATH)
    print(f"   Total images in dataset: {len(df)}")
    
    # Filter only melanoma and benign classes
    print("\n[2/6] Filtering melanoma and benign classes...")
    df_filtered = df[df['dx'].isin(['mel', 'nv'])].copy()
    print(f"   Melanoma images: {len(df_filtered[df_filtered['dx'] == 'mel'])}")
    print(f"   Benign images: {len(df_filtered[df_filtered['dx'] == 'nv'])}")
    
    # Balance the dataset (equal samples from each class)
    print("\n[3/6] Balancing dataset...")
    mel_df = df_filtered[df_filtered['dx'] == 'mel']
    nv_df = df_filtered[df_filtered['dx'] == 'nv']
    
    samples_per_class = SAMPLE_SIZE // 2
    
    # Sample equal amounts from each class
    mel_sample = mel_df.sample(n=min(samples_per_class, len(mel_df)), random_state=RANDOM_SEED)
    nv_sample = nv_df.sample(n=min(samples_per_class, len(nv_df)), random_state=RANDOM_SEED)
    
    df_balanced = pd.concat([mel_sample, nv_sample]).reset_index(drop=True)
    print(f"   Sampled {len(df_balanced)} images ({len(mel_sample)} malignant, {len(nv_sample)} benign)")
    
    # Map to binary classes
    df_balanced['class'] = df_balanced['dx'].map(CLASS_MAPPING)
    
    # Split dataset: 70% train, 15% validation, 15% test
    print("\n[4/6] Splitting dataset...")
    train_df, temp_df = train_test_split(df_balanced, test_size=0.3, 
                                          stratify=df_balanced['class'], 
                                          random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, 
                                        stratify=temp_df['class'], 
                                        random_state=RANDOM_SEED)
    
    print(f"   Training set: {len(train_df)} images")
    print(f"   Validation set: {len(val_df)} images")
    print(f"   Test set: {len(test_df)} images")
    
    # Create directory structure
    print("\n[5/6] Creating directory structure...")
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    for split_name in splits.keys():
        for class_name in ['benign', 'malignant']:
            dir_path = os.path.join(OUTPUT_DIR, split_name, class_name)
            os.makedirs(dir_path, exist_ok=True)
    
    print("   Directory structure created!")
    
    # Copy images to respective folders
    print("\n[6/6] Copying images...")
    total_copied = 0
    total_missing = 0
    
    for split_name, split_df in splits.items():
        print(f"\n   Processing {split_name} set...")
        copied = 0
        missing = 0
        
        for _, row in split_df.iterrows():
            image_id = row['image_id']
            class_name = row['class']
            
            # Find source image
            src_path = find_image_path(image_id, IMAGE_FOLDERS)
            
            if src_path:
                # Destination path
                dst_path = os.path.join(OUTPUT_DIR, split_name, class_name, f"{image_id}.jpg")
                shutil.copy2(src_path, dst_path)
                copied += 1
            else:
                missing += 1
                print(f"   Warning: Image not found - {image_id}.jpg")
        
        print(f"   ✓ Copied: {copied} images")
        if missing > 0:
            print(f"   ✗ Missing: {missing} images")
        
        total_copied += copied
        total_missing += missing
    
    # Summary
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE!")
    print("=" * 60)
    print(f"Total images copied: {total_copied}")
    print(f"Total images missing: {total_missing}")
    print(f"\nDataset saved in: {OUTPUT_DIR}/")
    print("\nYou can now proceed to train_model.py")
    print("=" * 60)

if __name__ == "__main__":
    try:
        prepare_dataset()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nMake sure you have:")
        print("1. HAM10000_metadata.csv in the current directory")
        print("2. HAM10000_images_part_1/ folder with images")
        print("3. HAM10000_images_part_2/ folder with images")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()