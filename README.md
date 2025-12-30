# Bayesian Neural Network for CHD Risk Prediction

This repository contains an implementation of a Bayesian Neural Network (BNN) for predicting 10-year coronary heart disease (CHD) risk using the Framingham Heart Study dataset.

## Dataset
The Framingham dataset includes demographic, behavioral, and medical risk factors for over 4,000 patients.  
The target variable is `TenYearCHD`, a binary indicator of whether a patient develops CHD within 10 years.

## Methodology
- Bayesian Neural Network implemented in PyTorch
- Variational inference (Bayes by Backprop) to learn posterior distributions over weights
- Bernoulli likelihood for binary classification
- Monte Carlo sampling at prediction time to estimate predictive uncertainty

## Evaluation
- ROC–AUC ≈ 0.69 on the test set
- Accuracy ≈ 0.65
- Predictive uncertainty quantified using 95% credible intervals
- Uncertain cases identified where the credible interval crosses the 0.5 decision threshold

## Motivation
Unlike deterministic neural networks, the Bayesian approach provides uncertainty-aware predictions, which are especially important in medical risk assessment and decision-making.

## Files
- `bnn_framingham.ipynb`: Colab notebook with full implementation
- `bnn_framingham.py`: Script version of the model
- `framingham.csv.xls`: Dataset used for training and evaluation
