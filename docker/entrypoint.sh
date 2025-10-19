#!/bin/bash

# Define source directories
SOURCE_LEISAAC="$LEISAAC_PATH/source/leisaac"

# Function to install a package
install_pkg() {
    local SOURCE_DIR=$1
    local PACKAGE_NAME=$(basename "$SOURCE_DIR")

    if [ -d "$SOURCE_DIR" ]; then
        # Check if pyproject.toml exists
        if [ -f "$SOURCE_DIR/pyproject.toml" ]; then
            echo "Installing $PACKAGE_NAME from $SOURCE_DIR"
            cd "$SOURCE_DIR" || { echo "Failed to enter directory: $SOURCE_DIR"; exit 1; }
            ${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip install -e . || { echo "Failed to install $PACKAGE_NAME"; exit 1; }
            cd - > /dev/null
            echo "$PACKAGE_NAME installed successfully."
        else
            echo "pyproject.toml not found in $SOURCE_DIR. Skipping."
        fi
    else
        echo "Source directory $SOURCE_DIR does not exist. Skipping."
    fi
}

# Install leisaac package
install_pkg "$SOURCE_LEISAAC"

echo "All tasks completed successfully."

# Execute any additional commands provided to the container
exec "$@"