# FRP Bar–Concrete Bond Strength Predictor

A local web application for research-oriented prediction and comparison of FRP bar–concrete bond strength. The application runs on a Windows computer with Python 3.11 and serves desktop and mobile interfaces over a local network.

## Overview

This repository contains the FRP Bond Strength Predictor v1.0, including its Python HTTP server, browser interfaces, model-training script and the 195-row dataset reconstructed from Appendix A of the supplied thesis PDF. The app currently trains nine regression models locally when launched for the first time, then loads the generated model files to serve predictions.

The thesis text reports 193 valid samples after outlier removal, while the included CSV contains 195 rows recoverable from Appendix A. The included dataset therefore does not reproduce the reported 193-row cleaned dataset exactly. Review this distinction before using or citing the results.

## Key Features

- Responsive English-language desktop and mobile interfaces.
- Nine selectable regression models: CatBoost, XGBoost, Gradient Boosting Decision Tree (GBDT), Random Forest, back-propagation neural network (BPNN), support vector regression (SVR), AdaBoost, multiple linear regression (MLR), and LASSO.
- Local HTTP prediction API with health and model-information endpoints.
- Input-range checks that flag predictions involving extrapolation beyond the ranges encoded in the application.
- Windows launcher that creates a Python 3.11 virtual environment, installs the pinned dependencies, trains the models on first launch and starts the local web server.
- Mobile access over the same Wi-Fi/LAN as the host computer.

## Repository Structure

```text
.
├── app.py                 # Local HTTP server and prediction API
├── train_models.py        # Trains and saves the nine regression models
├── requirements.txt       # Pinned Python dependencies
├── start_windows.bat      # Windows setup, training and launch script
├── repair_windows.bat     # Rebuilds the local environment and models
├── START_HERE.txt         # Quick-start instructions
├── data/
│   └── frp_dataset_appendixA_195.csv
└── static/
    ├── index.html         # Desktop interface
    ├── mobile.html        # Mobile interface
    ├── manifest.webmanifest
    └── app-icon*.png
```

Generated model files are written to `models/` when training runs and are intentionally excluded from version control. They can be regenerated from the included training script and dataset.

## Requirements

- Windows 10 or Windows 11, 64-bit
- 64-bit Python 3.11.x
- Internet access on first launch to install the pinned Python packages

## Run Locally on Windows

1. Install Python 3.11.x (64-bit).
2. Download or clone this repository.
3. Double-click `start_windows.bat`.
4. On first launch, allow the script to create `.venv`, install the dependencies and train the nine models. This may take several minutes.
5. The desktop interface opens at `http://127.0.0.1:8765/`.
6. To use the mobile interface, connect the phone and computer to the same Wi-Fi/LAN and open the mobile URL printed in the command window. Keep the computer on and the server running while using the phone.

If Windows Firewall prompts you, allow access on a private network only if you want other devices on your local network to reach the app.

## API Endpoints

- `GET /api/health` — reports service health and Python version.
- `GET /api/info` — returns model names, feature names, input ranges, reported metrics and local access URLs.
- `POST /api/predict` — predicts bond strength. Submit JSON containing a `model` name and a `features` object with all 13 feature keys listed in `train_models.py`.

The API is intended for local research use. The current launcher binds the server to `0.0.0.0` to support devices on the same LAN; do not expose this development server directly to the public internet.

## Deployment Notes

Pushing this repository to GitHub publishes the source code; it does not make the Python prediction service run online. GitHub Pages can host static files but cannot run this Python backend. Public online access requires deployment to a Python-capable application host and appropriate production security, networking and model-file configuration. The included Windows launcher is designed for local use.

## Intended Use and Limitations

This tool is intended solely for theoretical analysis, research support and parametric comparison of FRP bar–concrete bond strength. Predictions must not replace design codes, structural design checks, construction acceptance, safety assessment or professional engineering judgement. Inputs outside the dataset ranges are extrapolations and should be interpreted with caution.

## Acknowledgements

Developed by Z Lab. The source package identifies the application as © 2026 PhD Hong. All rights reserved. Confirm that you have the right to redistribute the included dataset and application materials before making this repository public.
