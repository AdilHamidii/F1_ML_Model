# Formula 1 Race Position Prediction (Machine Learning)

This repository contains an **applied machine learning project** that predicts **Formula 1 race finishing positions** using **only pre-race information**.  
The goal is to demonstrate **clean data engineering**, **sound ML methodology**, and **realistic evaluation on noisy real-world data**.

---

## Objective

Predict a driver’s **race finishing position** based on:
- Qualifying lap times (Q1, Q2, Q3)
- Qualifying position
- Team
- Season year

Race results are used **only as training labels**, never as model inputs.

---

## Approach

- Data collected using the **FastF1** library (qualifying + race sessions)
- One row per driver per race
- DNFs / DSQs / DNS handled as missing race positions
- Lap times converted to seconds for numerical modeling

### Model
- **Algorithm:** Random Forest Regressor
- **Framework:** scikit-learn
- **Preprocessing:**
  - One-hot encoding for teams
  - Numerical passthrough for lap times and positions
- **Sample weighting:**
  - More importance to recent seasons
  - Higher weight for front-row qualifiers

### Evaluation strategy
- **Train:** 2022–2023 seasons  
- **Test:** 2024 season  
This prevents data leakage and mirrors real-world forecasting.

---

## Results

Typical performance on the 2024 season:
- **Mean Absolute Error:** ~3.1 positions
- **±3 positions accuracy:** ~53%

Given the inherent randomness of F1 races, this represents a **strong baseline** using only pre-race data.
might be a sign for me to start gambling 

---

## What This Demonstrates

- End-to-end ML workflow (data → model → evaluation)
- Handling imperfect, real-world sports data
- Correct temporal validation
- Practical use of scikit-learn pipelines
- Domain-informed modeling decisions

---

## Technologies

- Python
- NumPy
- pandas
- scikit-learn
- FastF1

---

## References

- scikit-learn documentation: https://scikit-learn.org/stable/
- NumPy documentation: https://numpy.org/doc/
- FastF1 documentation: https://theoehrly.github.io/Fast-F1/
- Freecode camp course : https://www.youtube.com/watch?v=0B5eIE_1vpU

---

