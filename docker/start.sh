#!/bin/bash

python -m imcui.api.server --config imcui/api/config/api.yaml &

python app.py --server_port=7860 &

tail -f /dev/null
