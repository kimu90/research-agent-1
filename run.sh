#!/bin/bash
python init_db.py
streamlit run main.py --server.address=0.0.0.0