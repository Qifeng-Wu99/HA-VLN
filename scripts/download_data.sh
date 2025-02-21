#!/bin/bash

# Set the dataset download directory
DATASET_DIR="Data"
mkdir -p $DATASET_DIR

download_and_extract() {
    local url=$1
    local output_file="$DATASET_DIR/$2"
    local extract_dir="$DATASET_DIR/$3"

    if [ -f "$output_file" ]; then
        echo "File already exists: $output_file, skipping download."
    else
        echo "Downloading $output_file ..."
        wget -c -O "$output_file" "$url"
    fi

    echo "Extracting $output_file ..."
    unzip -q -o "$output_file" -d "$DATASET_DIR"

    rm -f "$output_file"
    echo "Deleted: $output_file after extraction."
}

# Download HAPS 2.0 dataset
download_and_extract "https://www.dropbox.com/scl/fo/6ofhh9vw5h21is38ahhgc/AKcBDbCk24N_OUYfh3k7nKI/HAPS2.0?rlkey=v88np78ugr49z3sqisnvo6a9i&subfolder_nav_tracking=1&dl=0" \
                     "HAPS2_0.zip" "HAPS2.0"

# Download HA-R2R-CE dataset
download_and_extract "https://www.dropbox.com/scl/fo/6ofhh9vw5h21is38ahhgc/AM5yiojqqr4t_XR7FsxMXFY/HA-R2R?rlkey=gvvqy4lsusthzwt9974kkyn7s&subfolder_nav_tracking=1&dl=0" \
                     "HAR2R-CE.zip" "HA-R2R"

echo "🎉 All datasets have been ready!"