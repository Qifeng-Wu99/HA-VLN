#!/bin/bash

# Ensure gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "gdown is not installed. Installing..."
    pip install gdown
fi

# Set the dataset download directory
DATASET_DIR="Data"
mkdir -p $DATASET_DIR

# Function to download and extract a ZIP file
download_and_extract() {
    local file_id=$1
    local output_file="$DATASET_DIR/$2"
    local extract_dir="$DATASET_DIR/$3"

    if [ -f "$output_file" ]; then
        echo "File already exists: $output_file, skipping download."
    else
        echo "Downloading $output_file from Google Drive..."
        gdown "https://drive.google.com/uc?id=$file_id" -O "$output_file"
    fi

    # Check if the file was downloaded successfully
    if [ ! -f "$output_file" ]; then
        echo "Download failed: $output_file"
        exit 1
    fi

    echo "Extracting $output_file to $extract_dir ..."
    mkdir -p "$extract_dir"
    unzip -q -o "$output_file" -d "$extract_dir"

    # Remove the ZIP file after extraction
    rm -f "$output_file"
    echo "Deleted: $output_file after extraction."
}

HAPS2_0_ID="1gNdA4_mDAhW6g6Yedq2ypOR1N1sFpARB"
HAR2R_CE_ID="1_-5StHsRP6REKrANMxKIscPtEP7sx-q0"

# Download and extract HAPS 2.0 dataset
download_and_extract "$HAPS2_0_ID" "HAPS2_0.zip" "HAPS2_0"

mv Data/HAPS2_0/human_motion_glbs_v3/* Data/HAPS2_0/
rmdir Data/HAPS2_0/human_motion_glbs_v3

# Download and extract HA-R2R-CE dataset
download_and_extract "$HAR2R_CE_ID" "HAR2R-CE.zip" "HA-R2R"

echo "🎉 All datasets are ready!"