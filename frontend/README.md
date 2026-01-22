# MVM² Frontend

A lightweight frontend to interact with the MVM² verification backend.

## Setup

1.  Ensure the Backend is running on port 8000:
    ```bash
    cd ..
    python backend/main.py
    ```

2.  Serve this frontend. You can use any static file server.
    
    Python (simplest):
    ```bash
    python -m http.server 3000
    ```
    
    Node/NPM:
    ```bash
    npx serve .
    ```

3.  Open browser to `http://localhost:3000`.

## Features
- Switch between **Text Input** and **Image Upload**.
- Calls `POST /solve/text` and `POST /solve/image`.
- Displays Teacher Explanation.
- Visualizes Consensus/Hallucination risks per step.
