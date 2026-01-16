## Purpose
This project was created to be submitted as Capstone Project for the [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) (2025 cohort).

## Problem Description
Production lines in manufacturing facilities make use of monitoring systems to detect potential issues and failures as early as possible so that they can be fixed as quickly as possible. Such monitoring systems typically use numerous sensors to monitor the different production steps. This usually results in large amounts of sensor data being collected. It then becomes a challenge to understand which sensor signals within the collected data are the most useful for identifying potential issues in the production line that require human intervention. This is where machine learning algorithms can help identify the most relevant sensor signals, as well as help build models that accurately identify when issues that require fixing occur.

## Project Goal
This project analyses real monitoring data collected from a semiconductor production line. The goal is to identify which are the most relevant signals within the many measured sensor signals, and based on those signals, train a machine learning model that accurately predicts the pass or fail of production line tests.
It is worth mentioning that the primary goal of this project is to demonstrate the steps involved in training and deploying a machine learning model, not to obtain the best performing model.

## Data Description
The data used for this project is the SECOM dataset, which is provided on the following page: https://doi.org/10.24432/C54305

The dataset contains 1567 sets of measurements with 590 features (i.e. measured signals). Out of the 1567 measurements, the test outcome is a fail in 104 cases.

The data provided on the page indicated above consists of a zip file that contains 3 files:
- secom.data: contains the selection of measured data (1567 rows and 590 columns).
- secom.names: includes information regarding the dataset.
- secom_labels.data: contains the labels that represent the outcome of the line testing for each measurement (–1 corresponds to a pass and 1 corresponds to a fail), along with a datetimestamp.

## Data Cleaning

## Model Selection and Training

## Model Deployment

## How to Reproduce this Project

## Note Regarding the Use of AI
Some of the code in this project was written with the assistance of AI through VS Code Copilot, with review by a Human (me). The rest of work involved in this project, such as problem selection, design decisions, package and tool selection, and writing of the README, was implemented by a Human (me).

## References
McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54305.