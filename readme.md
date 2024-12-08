# Phishing URL Detection

## Overview

The Phishing URL Detection project aims to identify potentially harmful URLs using machine learning techniques. By analyzing various features of a URL, the system can classify it as either "Safe" or "Unsafe." This project leverages multiple machine learning models to achieve high accuracy in detecting phishing attempts.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)

## Features

- URL analysis using various features such as:
  - Length of the URL
  - Presence of special characters
  - Use of HTTPS
  - Domain registration length
  - Favicon presence
  - And many more...
- Multiple machine learning models including:
  - Logistic Regression
  - K-Nearest Neighbors
  - Support Vector Machine
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - CatBoost
  - XGBoost
  - Multi-layer Perceptron
- Web interface built with Flask for user interaction.

## Technologies Used

- Python
- Flask
- Scikit-learn
- NumPy
- Pandas
- Requests
- Whois
- HTML/CSS (Bootstrap)

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/Lava-Kumar-PL/Phishing_url_detection.git
   ```

2. Change the directory:

   ```bash
   cd phishing-url-detection
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Flask application:

   ```bash
   python app.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000/`.

3. Enter a URL in the input field and click "Analyze" to see the results.

## Data

The dataset used for training the models is sourced from Kaggle. It contains over 11,000 samples with 30 features related to URLs and a class label indicating whether the URL is phishing or not.

## Result

<img src="https://github.com/Lava-Kumar-PL/Phishing_url_detection/blob/main/results/phishingGIF.gif " width="400" />

<img src="https://github.com/Lava-Kumar-PL/Phishing_url_detection/blob/main/results/url.png " width="400" />

<img src="https://github.com/Lava-Kumar-PL/Phishing_url_detection/blob/main/results/prediction.png " width="400" />
