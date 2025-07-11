# GTAV Fingerprint Solver

This tool helps you solve the fingerprint hacking minigame in GTA V by automatically detecting and matching fingerprint segments.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AdvaitKale01/gta5-fingerprint-solver.git
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```bash
   python main.py
   ```

## Usage

- Press `F10` to detect fingerprints while the app is running.
- A new window will open on your second monitor (or your main monitor if you only have one), displaying the detected answers in green boxes.

## Setting Up Fingerprint Detection Bounds

If you want to reconfigure the fingerprint detection regions:

1. Delete the `fingerprint_regions.json` file from the project directory.
2. The next time you run the app, you will be prompted to set up the fingerprint detection bounds interactively.

## Notes
- Make sure GTA V is running and the fingerprint minigame is visible on your screen when you press `F10`.
- The tool works best with the default game resolution and UI layout.

---

Feel free to open issues or pull requests for improvements or bug fixes!
