#!/bin/bash
# Run the test.py script with the virtual environment
cd "$(dirname "$0")"
source venv/bin/activate
python3 test.py "$@"

