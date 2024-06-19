### Repository Summary: Mushroom Classification Analysis

This repository contains the R coding solution and the final report for a project focused on classifying mushrooms as edible or poisonous based on their visual and olfactory characteristics. The analysis employs various machine learning models, including logistic regression, decision trees, and random forests, to determine the best predictive model. The report, available in both PDF and HTML formats, includes detailed model evaluations, tuning procedures, and cross-validation results.

### Project Overview

**Task Description:**
The project aims to predict the edibility of mushrooms using a dataset from The Audubon Society Field Guide to North American Mushrooms. The dataset includes categorical attributes such as CapShape, CapSurface, CapColor, Odor, and Height. The goal is to identify the most effective machine learning model for predicting whether a mushroom is edible or poisonous.

**Analysis Steps:**

1. **Data Preparation:**
   - Load and preprocess the dataset `mushrooms.csv`, ensuring all categorical attributes are properly encoded.

2. **Model Selection and Tuning:**
   - **Logistic Regression:** Fit a logistic regression model using all attributes. Tune the model for optimal performance.
   - **Decision Trees:** Build a decision tree classifier and tune its parameters to improve classification accuracy.
   - **Random Forests:** Train a random forest model, optimizing the number of trees and other hyperparameters for the best results.

3. **Cross-Validation:**
   - Perform cross-validation on each model to evaluate its predictive performance.
   - Use the number of correctly classified mushrooms as the primary metric for model evaluation.
   - Apply statistical tests to compare the performance of different models and determine if the differences are statistically significant.

4. **Results Presentation:**
   - Summarize the tuning process and the final parameters for each model.
   - Present cross-validation results, including accuracy scores and statistical significance tests.
   - Provide visualizations and tables to support the findings and make the results easily interpretable.

**Results:**
The report includes a comprehensive comparison of logistic regression, decision trees, and random forests for mushroom classification. It discusses the tuning process, cross-validation results, and the statistical significance of the performance differences between models. The final report highlights the most effective model for predicting mushroom edibility based on the given attributes.

**Files:**
- **R Scripts:** Contains the R code for data loading, preprocessing, model training, tuning, and evaluation.
- **Data:** Includes the dataset `mushrooms.csv`.
- **Reports:** The final report in both PDF and HTML formats, documenting the analysis, results, and conclusions.

This project showcases the application of various machine learning techniques to a classic dataset, providing insights into the effectiveness of different models for classifying mushrooms as edible or poisonous.
