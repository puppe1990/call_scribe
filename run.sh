#!/bin/bash
# Run the init.py script with the virtual environment
cd "$(dirname "$0")"
source venv/bin/activate
python3 init.py "$@"

