#!/bin/bash

# Run the database initialization script
python /app/init_db.py

# Now run the Streamlit app
streamlit run /app/main.py --server.address=0.0.0.0
