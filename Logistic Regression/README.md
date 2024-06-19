### Repository Summary: Brexit Vote Analysis

This repository contains the R coding solution and the final report for a project that statistically analyzes demographic data to understand voting patterns in the UK's 2016 EU referendum (Brexit). The analysis focuses on determining the influence of various demographic factors on the likelihood of an electoral ward voting to Leave the EU. The report, available in both PDF and HTML formats, includes comprehensive statistical analysis, visualizations, and detailed interpretations.

### Project Overview

**Task Description:**
The project aims to analyze the Brexit vote using demographic data from electoral wards across the UK. The goal is to identify which demographic factors had the most significant impact on whether a ward voted to Leave or Remain in the EU. The tasks involve building logistic regression models, interpreting the coefficients to assess the relevance and impact of each demographic variable, and comparing the findings with existing visualizations from the Guardian newspaper.

**Analysis Steps:**

1. **Logistic Regression Model:**
   - **Objective:** To identify which demographic factors are significant in predicting the Brexit vote.
   - **Inputs:** Proportion of ABC1 social class (abc1), median income (medianIncome), median age (medianAge), proportion with higher education (withHigherEd), and proportion not born in the UK (notBornUK).
   - **Output:** Binary variable indicating whether the ward voted for Brexit (voteBrexit).

2. **Model Interpretation:**
   - **Coefficients Analysis:** Determine the direction and magnitude of the effect of each input variable.
   - **Significance Assessment:** Identify which inputs have strong effects and order them by decreasing relevance.
   - **Comparison with Guardian Visuals:** Compare the model's findings with the demographic trends presented by the Guardian newspaper.

3. **Factors Affecting Interpretability:**
   - **Discussion:** Consider multicollinearity, variable scaling, and other factors that might affect the interpretation of regression coefficients.
   - **Reliability Assessment:** Evaluate the reliability of determining relevant inputs and order them based on their influence.

4. **Alternative Analysis Approach:**
   - **Presentation:** Introduce an alternative method for analyzing the data (e.g., stepwise regression, regularization techniques).
   - **Implementation:** Carry out the alternative analysis and discuss its benefits and drawbacks compared to the initial approach.

**Results:**
The report includes the results of the logistic regression analysis, detailed interpretations of the model coefficients, and a comparison with the Guardian's demographic trends. It also discusses the factors affecting the interpretability of the model and presents an alternative analysis approach, highlighting its advantages and disadvantages.

**Files:**
- **R Scripts:** Contains the R code used for data analysis, model fitting, and result visualization.
- **Data:** Includes the dataset `brexit.csv`.
- **Reports:** The final report in both PDF and HTML formats, documenting the analysis, results, and conclusions.

This project provides a thorough statistical investigation into the demographic factors influencing the Brexit vote, offering valuable insights into the relationship between social and economic variables and electoral outcomes.
