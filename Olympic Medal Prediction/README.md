### Repository Summary: Olympic Medal Prediction Analysis

This repository contains the R coding solution and the final report for a project aimed at predicting Olympic medal counts based on a country's population and GDP. The analysis utilizes linear regression models, log-transformed inputs, and clustering algorithms to investigate the relationship between these variables and the number of medals won in the 2012 and 2016 Olympic Games. The report includes detailed results, visualizations, and model evaluations in both PDF and HTML formats.

### Project Overview

**Task Description:**
The project examines how the number of Olympic medals won by a country can be predicted using national population and GDP data. The analysis spans the Olympic Games held in 2008, 2012, and 2016, focusing on 71 countries that won at least one gold medal in each of these games. The tasks involve creating and assessing linear regression models, exploring the impact of log-transformation, employing K-means clustering, and evaluating the probability of winning at least one medal.

**Analysis Steps:**
1. **Linear Regression Model:** 
   - Using population and GDP to predict the medal count in the 2012 Olympics.
   - Assessing model performance by predicting the medal count in the 2016 Olympics.

2. **Log-Transformation:**
   - Applying log-transformation to population and GDP inputs.
   - Comparing model performance between raw and log-transformed inputs.

3. **Clustering with K-means:**
   - Clustering log-transformed inputs using the K-means algorithm.
   - Training separate linear regression models for each cluster.
   - Validating the optimal number of clusters and combining predictions.

4. **Probability Estimation:**
   - Calculating the probability of a country winning at least one medal using model parameters.
   - Using the UK as a case study for this estimation.

5. **Model Selection:**
   - Comparing models from tasks 1 to 3.
   - Selecting the best model for accurate medal count prediction and justifying the choice.

**Results:**
The analysis provides insights into the effectiveness of different modeling approaches and transformations. It discusses the benefits of log-transformation and clustering, and evaluates the models based on their predictive accuracy. The final report includes comprehensive tables, figures, and discussions to support the findings.

**Files:**
- **R Scripts:** Contains the R code used for data analysis and model training.
- **Data:** Includes the dataset `medal_pop_gdp_data_statlearn.csv`.
- **Reports:** The final report in both PDF and HTML formats, documenting the analysis, results, and conclusions.

This project showcases a thorough investigation into predicting Olympic medal counts using statistical modeling techniques, highlighting the importance of data transformation and clustering in enhancing predictive performance.
