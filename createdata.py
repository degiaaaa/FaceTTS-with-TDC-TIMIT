import os
import shutil
import zipfile
import random

def create_folder_structure(base_path):
    # Define the folder names
    folders = ['mp4', 'wav', 'trainval']
    
    # Create the base path if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    # Create each folder in the base path
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")

def extract_files_from_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted all contents of {zip_path} to {extract_to}")

def copy_and_rename_files(source_dir, dest_dir, prefix, files_to_copy):
    # Adjust the source directory to the deeper nested folder if needed
    nested_dir = os.path.join(source_dir, 'straightcam')
    if os.path.exists(nested_dir):
        source_dir = nested_dir

    for file_name in files_to_copy:
        src_path = os.path.join(source_dir, file_name)
        if os.path.exists(src_path):
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            dest_path = os.path.join(dest_dir, file_name)
            shutil.copy(src_path, dest_path)
            print(f"Copied {src_path} to {dest_path}")
        else:
            print(f"Source file does not exist: {src_path}")

# Define the base paths
base_path_lrs3 = r"C:\Users\debor\OneDrive\Desktop\TCDTIMIT\data\lrs3"
trainval_path = os.path.join(base_path_lrs3, 'trainval')

# Create the folder structure for the mp4, wav, and trainval directories
create_folder_structure(base_path_lrs3)

# Define the source directory containing the volunteers
source_base_path = r"D:\TCDTimit Corpus Database\volunteers"
folders = [
    '01M', '02M', '03F', '04M', '05F', '06M', '07F', '08F', '09F', '10M',
    '11F', '12M', '14M', '16M', '17F', '18M', '19M', '20M', '22M', '26M',
    '32F', '33F', '38F', '39M', '40F', '41M', '42M', '44F', '45F', '46F',
    '48M', '49F', '50F', '52M', '56M', '58F', '59F', '13F', '15F', '21M',
    '23M', '24M', '25M', '28M', '29M', '30F', '31F', '34M', '36F', '37F',
    '43F', '47M', '51F', '54M', '57M' 
]

# Set a random seed for reproducibility
random.seed(42)

# Shuffle the list for random split
random.shuffle(folders)

# Calculate split indices
num_folders = len(folders)
num_val = int(0.1 * num_folders)
num_test = int(0.2 * num_folders)
num_train = num_folders - num_val - num_test

# Split the folders
val_folders = folders[:num_val]
test_folders = folders[num_val:num_val+num_test]
train_folders = folders[num_val+num_test:]

# Define the files to extract and copy
files_to_extract = ['sa1.mp4', 'SA1.txt', 'sa1.wav']

# Helper function to copy files to the correct directories
def process_files(folders, split_name):
    for folder in folders:
        zip_path = os.path.join(source_base_path, folder, 'straightcam.zip')
        extract_to = os.path.join(source_base_path, folder)
        extract_files_from_zip(zip_path, extract_to)

        # Define destination directories
        dest_dir_mp4 = os.path.join(base_path_lrs3, 'mp4', split_name, folder)
        dest_dir_wav = os.path.join(base_path_lrs3, 'wav', split_name, folder)

        # Copy and rename files
        copy_and_rename_files(extract_to, dest_dir_mp4, folder, ['sa1.mp4'])
        copy_and_rename_files(extract_to, dest_dir_wav, folder, ['sa1.wav'])

        # Copy files to trainval directory and mp4/trainval directory if split is train or val
        if split_name in ['train', 'val']:
            dest_dir_trainval = os.path.join(trainval_path, folder)
            copy_and_rename_files(extract_to, dest_dir_trainval, folder, files_to_extract)
            
            dest_dir_mp4_trainval = os.path.join(base_path_lrs3, 'mp4', 'trainval', folder)
            copy_and_rename_files(extract_to, dest_dir_mp4_trainval, folder, ['sa1.mp4'])

# Process validation set
process_files(val_folders, 'val')

# Process test set
process_files(test_folders, 'test')

# Process train set
process_files(train_folders, 'train')
