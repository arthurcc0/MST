#!/bin/bash

# Sequential Python Script Runner
# This script executes two Python scripts one after another

# Define the Python scripts to run
SCRIPT1="step2a_calc_sub.py"
SCRIPT2="step2b_crop_or_pad.py"

echo "Starting sequential execution of Python scripts..."

# Execute first Python script
echo "Running $SCRIPT1..."
python3 "$SCRIPT1"

# Check if first script executed successfully
if [ $? -eq 0 ]; then
    echo "$SCRIPT1 completed successfully."
    
    # Execute second Python script
    echo "Running $SCRIPT2..."
    python3 "$SCRIPT2"
    
    # Check if second script executed successfully
    if [ $? -eq 0 ]; then
        echo "$SCRIPT2 completed successfully."
        echo "All scripts executed successfully!"
    else
        echo "Error: $SCRIPT2 failed with exit code $?"
        exit 1
    fi
else
    echo "Error: $SCRIPT1 failed with exit code $?"
    echo "Skipping $SCRIPT2 due to previous failure."
    exit 1
fi

echo "Script execution completed."