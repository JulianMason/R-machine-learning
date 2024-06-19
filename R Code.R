knitr::opts_chunk$set(echo=TRUE)

# import data
data <- read.csv("medal_pop_gdp_data_statlearn.csv")
data$Country["Great Britain"]

library(knitr)
library(kableExtra)
library(dplyr)
library(ggplot2)
library(tidyr)
#install.packages("caret")
#install.packages("rmarkdown")
library("caret")
library("rmarkdown")

################################################## TASK 1 ###############################################################

medals_2012 <- data[, c(2, 3, 5)] # SUB DATA
medal_model = glm(formula = Medal2012 ~ GDP + Population, data=medals_2012) # GLM MODEL

coef_table <- summary(medal_model)$coefficients

# CREATES TABLE OF SUMMARY STATISTICS
coef <- data.frame(
  Estimate = c(coef_table[1, 1], coef_table[2, 1], coef_table[3, 1]),
  Standard_Error = c(coef_table[1, 2], coef_table[2, 2], coef_table[3, 2]),
  t_value = c(coef_table[1, 3], coef_table[2, 3], coef_table[3, 3]),
  p_value = c(coef_table[1, 4], coef_table[2, 4], coef_table[3, 4])
)
rownames(coef) <- c("Intercept", "GDP", "Population")

# CI
#n = length(medal_model$Medal2012)
#df = n-3
#tc = qt(p=0.975, df=df) # t-statistic
#print(summary(medal_model)$coefficients[1, 1])
#confint(medal_model)

inter1 = c(confint(medal_model)[1,1], confint(medal_model)[1,2])

gdp_ci = c(confint(medal_model)[2,1], confint(medal_model)[2,2])

pop_ci = c(confint(medal_model)[3,1], confint(medal_model)[3,2])


#confint(medal_model)

# CREATES TABLE FOR CONFIDENCE INTERVALS
confin <- data.frame(
  " " = c("Intercept", "GDP", "Population"),
  "Confidence_Interval" = c("3.14, 9.02", "6.13e-3, 8.90e-3", "-8.85e-9, 1.93e-8"),
  "Estimate_in_Interval" = c("Yes", "Yes", "Yes")
) # CREATES DATAFRAME OF CONFIDENCE INTERVALS

kable(
  confin, 
  caption = "Table 1. Estimates and their confidence intervals",
  align = c("l", "c", "c", "c", "c", "c"),
  linesep = ""
) %>% 
  kable_styling(
    full_width = FALSE, 
    position = "left", 
    bootstrap_options = c("striped", "hover", "condensed", "responsive"),
  ) %>% 
  column_spec(1, bold = TRUE)

ac_pred_df <- data[, c(1,2, 3, 6)]

test_df <- ac_pred_df[, c(2, 3)]
ac_pred_df$Pred2016 <- ceiling(predict(medal_model, newdata=test_df))
ac_pred_df$diff <- ac_pred_df$Pred2016 - ac_pred_df$Medal2016 # RESIDUALS 

# Assessing Model Accuracy with Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
MSE <- mean(abs(ac_pred_df$diff))
RMSE <- sqrt(mean(ac_pred_df$diff^2))

# Visualizing Top 15
plotdata <- ac_pred_df[, c(1, 4, 5, 6)]
colnames(plotdata) <- c("Country","Actual","Predicted","Difference" )

plot_data <- data.frame(country = plotdata$Country,
                        actual = plotdata$Actual,
                        predicted = plotdata$Predicted
                        )

plot_data <- plot_data %>%
  arrange(desc(actual))%>%
  head(15)

plot_data <- plot_data %>%
  gather(type, count, actual:predicted)

ggplot(plot_data, aes(x = country, y = count, fill = type)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Figure 1. Actual vs. Predicted Medals Won",
       x = "Country",
       y = "Medals Won") +
  theme(legend.position = "top",
        axis.text.x = element_text(angle = 45, hjust = 1))

cat("The Root Mean Squared Error (RMSE) of Model 1 is ",round(RMSE, 2), "\n")


############################################## TASK 2 #######################################################

#Log Transformation
medal_model_2 = glm(formula = Medal2012 ~ log(GDP) + log(Population), data=medals_2012)

# Log prediction
ac_pred_df$Log_pred2016 <- ceiling(predict(medal_model_2, newdata=test_df))

#confint(medal_model_2)

# CREATES TABLE OF CONFIDENCE INTERVALS
confin <- data.frame(
  " " = c("Intercept", "log(GDP)", "log(Population)"),
  "Confidence_Interval" = c("-97.61, -1.12", "2.52, 8.57", "-1.53, 5.49"),
  "Estimate_in_Interval" = c("Yes", "Yes", "Yes")
)

kable(
  confin, 
  caption = "Table 2. Log estimates and their confidence intervals",
  align = c("l", "c", "c", "c", "c", "c"),
  linesep = ""
) %>% 
  kable_styling(
    full_width = FALSE, 
    position = "left", 
    bootstrap_options = c("striped", "hover", "condensed", "responsive"),
  ) %>% 
  column_spec(1, bold = TRUE)

log_residuals <- ac_pred_df$Medal2016 - exp(ac_pred_df$Log_pred2016)
log_mse = mean(log_residuals^2) # MSE
log_rmse = sqrt(log_mse) # RMSE

# CREATES COMPARISON TABLE OF MODEL METRICS BETWEEN LOG-TRANSFORMED AND ORIGINAL MODEL
comp <- data.frame(
  "Model" = c("Log-Transformed", "Original"),
  "Residual_Deviance" = c("17,310.00", "8,986.20"),
  "AIC" = c("599.70", "553.20"),
  "RMSE" = c("5.98e+17", "9.17")
)
kable(
  comp, 
  caption = "Table 3. Log-Transformed Model vs Original Model",
  align = c("l", "c", "c", "c", "c", "c")
) %>% 
  kable_styling(
    full_width = FALSE, 
    position = "left", 
    bootstrap_options = c("striped", "hover", "condensed", "responsive")
  ) %>% 
  column_spec(1, bold = TRUE)

#PLOTTING GRAPH OF BOTH MODEL PREDICTIONS
plotdata <- ac_pred_df[, c(1, 4, 5, 7)]
colnames(plotdata) <- c("Country","Actual","Original","Log_Transformed" )

plot_data <- data.frame(country = plotdata$Country,
                        actual = plotdata$Actual,
                        original = plotdata$Original,
                        log_predicted = plotdata$Log_Transformed)

plot_data <- plot_data %>%
  arrange(desc(actual))%>%
  head(15)

plot_data <- plot_data %>%
  gather(type, count, actual:log_predicted)

# PLOTTING THE ACTUAL MEDALS WON VS LOG TRANSFORMED PREDICTION
ggplot(plot_data, aes(x = country, y = log(count), fill = type)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Figure 2. Actual vs. Predicted vs Log_Transformed Medals Won",
       x = "Country",
       y = "Medals Won") +
  theme(legend.position = "top",
        axis.text.x = element_text(angle = 45, hjust = 1))


############################################## TASK 3 #######################################################

df <- data[, c(2, 3, 5, 6)]
df$log_GDP <- log(data$GDP)
df$log_Population <- log(data$Population)
colnames(df) <- c("GDP","Population", "Medal2012", "Medal2016", "log_GDP","log_Population")

# Partition data into training and test sets
trainIndex <- createDataPartition(df$Medal2012, p = 0.8, list = FALSE)
train <- df[trainIndex, ]
test <- df[-trainIndex, ]

log_vars <- c("log_GDP", "log_Population")
raw_vars <- c("GDP", "Population")
kmeans_df <- rbind(train[, log_vars], test[, log_vars])
kmax <- 10
wss <- sapply(1:kmax, function(k) {
  kmeans(kmeans_df, k, nstart = 10)$tot.withinss
})

# Determine optimal number of clusters using elbow method
plot(1:kmax, wss,
     type = "b", pch = 19, frame = FALSE, 
     xlab = "Number of clusters K",
     ylab = "Total within-clusters sum of squares",
     main= "Figure 3. Elbow Plot")
abline(v = 4, col = "red")

# Fit k-means clustering model with optimal number of clusters
k_opt <- 4
kmeans_model <- kmeans(kmeans_df, k_opt, nstart = 10)
train$cluster <- kmeans_model$cluster[trainIndex]
test$cluster <- kmeans_model$cluster[-trainIndex]

# Fit a separate generalized linear regression model for each cluster
models <- list()
for (i in 1:k_opt) {
sub_train <- train[train$cluster == i, c(raw_vars, "Medal2012")]
models[[i]] <- glm(Medal2012 ~ GDP + Population, data = sub_train, family = "gaussian")
}

# Make predictions on the test set
test$pred <- NA
for (i in 1:k_opt) {
test[test$cluster == i, "pred"] <- predict(models[[i]], newdata = test[test$cluster == i, ])
}

# Calculate RMSE
#test %>%
  #mutate(residuals = pred - Medal2012,
         #squared_residuals = residuals^2) %>%
  #summarise(RMSE = sqrt(mean(squared_residuals)))


medal_counts <- data.frame(Actual = test$Medal2016, Predicted = test$pred)
 # VISUALISING 
ggplot(test, aes(x = Medal2016, y = pred, color = factor(cluster))) +
  geom_point(shape = 21, size = 3, fill = "white") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(x = "Actual Medal Count (2016)", y = "Predicted Medal Count (2016)", title = "Figure 4. Actual Medal Count vs Cluster Predicted Medal Count") +
  scale_color_discrete(name = "Cluster")

log_dispersion <- summary(medal_model_2)$dispersion

############################################## TASK 4 #######################################################

gb <- ac_pred_df[ac_pred_df$Country == "Great Britain", "Log_pred2016"]

gb_prob <- 1 -pnorm(1, gb, sqrt(log_dispersion))

cat("The probability that Great Britain wins at least one medal is",round(gb_prob, 4), "\n")


############################################## TASK 5 #######################################################

winner_mse = rep(NA, 100)
winner_rmse = rep(NA, 100)
winner_ll = rep(NA, 100)

# Function to compute Euclidean distance between two points
  euclideanDistance <- function(x1, x2, y1, y2) {
    sqrt((x1 - x2) ^ 2 + (y1 - y2) ^ 2)
  }

# Function to allot a cluster to a datapoint (gdp, population) based on its Euclidean distance from the four centers
  allotCluster <- function(gdp, population, mean_gdp_c1, mean_pop_c1, mean_gdp_c2, mean_pop_c2, mean_gdp_c3, mean_pop_c3, mean_gdp_c4, mean_pop_c4) {
    distances <- c(euclideanDistance(gdp, mean_gdp_c1, population, mean_pop_c1),
                   euclideanDistance(gdp, mean_gdp_c2, population, mean_pop_c2),
                   euclideanDistance(gdp, mean_gdp_c3, population, mean_pop_c3),
                   euclideanDistance(gdp, mean_gdp_c4, population, mean_pop_c4))
    cluster <- which.min(distances)
    return(cluster)
  }


# For-loop to run 100 times 
for(i in 1:100){
# Partition data into training and test sets
  eval_df <- data[, c(1, 2, 3, 5, 6)]
  
  trainIndex <- createDataPartition(eval_df$Medal2012, p = 0.8, list = FALSE)
  train_data <- eval_df[trainIndex, ]
  test_data <- eval_df[-trainIndex, ]

##### TASK 1 MODEL #####
  task1_model <- glm(Medal2012 ~ GDP + Population, data=train_data) # Train model
  task1_pred <- predict(task1_model, newdata=test_data) # Test model
  
  task1_mse <- mean((test_data$Medal2016 - task1_pred)^2) # MSE
  task1_rmse <- sqrt(task1_mse) # RMSE
  
  task1_dispersion <- summary(task1_model)$dispersion
  task1_ll <- sum(dnorm(test_data$Medal2016, task1_pred, sqrt(task1_dispersion), log=TRUE))
  task1_ll # Predictive log likelihood of Task 1 Model
  
##### TASK 2 MODEL #####
  task2_model <- glm(Medal2012 ~ log(GDP) + log(Population), data=train_data) # Train model
  task2_pred <- predict(task2_model, newdata = test_data)
  
  task2_mse <- mean((test_data$Medal2016 - task2_pred)^2) # MSE of Task 2
  task2_rmse <- sqrt(task2_mse) # RMSE Task 2
  
  task2_dispersion <- summary(task2_model)$dispersion
  task2_ll <- sum(dnorm(test_data$Medal2016, task2_pred, sqrt(task2_dispersion), log=TRUE))
  
  
###### TASK 3 MODEL ######
  train_log <- data.frame(log_gdp = log(train_data$GDP), log_population = log(train_data$Population))
  test_log <- data.frame(log_gdp = log(test_data$GDP), log_population = log(test_data$Population), Medal2016 = test_data$Medal2016)
  
  train_clust <- kmeans(train_log, centers = 4, nstart=100)$cluster # Cluster inputs into 4 clusters
  train_data$cluster <- train_clust
  
  # Cluster 1 Training
  clust1 <- data.frame(train_data[train_data$cluster == 1, ])
  mean_gdp_c1 <- mean(log(clust1$GDP))
  mean_pop_c1 <- mean(log(clust1$Population))
  
  clust1_model <- glm(Medal2012 ~ GDP + Population, data=clust1)
  
  # Cluster 2 Training
  clust2 <- data.frame(train_data[train_data$cluster == 2, ])
  mean_gdp_c2 <- mean(log(clust2$GDP))
  mean_pop_c2 <- mean(log(clust2$Population))
  
  clust2_model <- glm(Medal2012 ~ GDP + Population, data=clust2) # Training Cluster 2
  
  # Cluster 3 Training
  clust3 <- data.frame(train_data[train_data$cluster == 3, ])
  mean_gdp_c3 <- mean(log(clust3$GDP))
  mean_pop_c3 <- mean(log(clust3$Population))
  
  clust3_model <- glm(Medal2012 ~ GDP + Population, data=clust3) # Training Cluster 3
  
  # Cluster 4 Training
  clust4 <- data.frame(train_data[train_data$cluster == 4, ])
  mean_gdp_c4 <- mean(log(clust4$GDP))
  mean_pop_c4 <- mean(log(clust4$Population))
  
  clust4_model <- glm(Medal2012 ~ GDP + Population, data=clust4) # Training Cluster 4
  
  # Allot cluster to each datapoint in test set
  test_log$cluster <- apply(test_log, 1, function(row){
    gdp <- as.numeric(row[["log_gdp"]])
    population <- as.numeric(row[["log_population"]])
    cluster <- allotCluster(gdp, population, mean_gdp_c1, mean_pop_c1, mean_gdp_c2, mean_pop_c2, mean_gdp_c3, mean_pop_c3, mean_gdp_c4, mean_pop_c4)
    return(cluster)
  })
  
  ## Cluster 1 Testing ##
  clust1_test <- data.frame(test_data[test_log$cluster == 1, ]) # Create test dataframe
  clust1_pred <- predict(clust1_model, newdata=clust1_test) # Predict cluster model using test data
  
  clust1_mse <- mean((clust1_test$Medal2016 - clust1_pred)^2, na.rm=TRUE) # Cluster 1 MSE
  clust1_rmse <- sqrt(clust1_mse)

  n1 <- nrow(clust1_test) # Number of records in cluster 1
  
  clust1_dispersion <- summary(clust1_model)$dispersion
  # Cluster 1 Log likelihood
  clust1_ll <- sum(dnorm(clust1_test$Medal2016, clust1_pred, sqrt(clust1_dispersion), log=TRUE))
  
  ## Cluster 2 Testing ##
  clust2_test <- data.frame(test_data[test_log$cluster == 2, ]) # Create test dataframe
  
  clust2_pred <- predict(clust2_model, newdata=clust2_test) # Predict cluster model using test data
  
  clust2_mse <- mean((clust2_test$Medal2016 - clust2_pred)^2, na.rm=TRUE) # Cluster 2 MSE
  clust2_rmse <- sqrt(clust2_mse)

  n2 <- nrow(clust2_test) # Number of records in cluster 2
  n2
  
  clust2_dispersion <- summary(clust2_model)$dispersion
  # Cluster 2 Log likelihood
  clust2_ll <- sum(dnorm(clust2_test$Medal2016, clust2_pred, sqrt(clust2_dispersion), log=TRUE))
  
  ## Cluster 3 Testing ##
  clust3_test <- data.frame(test_data[test_log$cluster == 3, ]) # Create test dataframe
  
  clust3_pred <- predict(clust3_model, newdata=clust3_test) # Predict cluster model using test data
  
  clust3_mse <- mean((clust3_test$Medal2016 - clust3_pred)^2, na.rm=TRUE) # Cluster 3 MSE
  clust3_rmse <- sqrt(clust3_mse)

  n3 <- nrow(clust3_test) # Number of records in cluster 3
  
  clust3_dispersion <- summary(clust3_model)$dispersion
  # Cluster 3 Log likelihood
  clust3_ll <- sum(dnorm(clust3_test$Medal2016, clust3_pred, sqrt(clust3_dispersion), log=TRUE))
  
  
  ## Cluster 4 Testing ##
  clust4_test <- data.frame(test_data[test_log$cluster == 4, ]) # Create test dataframe
  
  clust4_pred <- predict(clust4_model, newdata=clust4_test) # Predict cluster model using test data
  
  clust4_mse <- mean((clust4_test$Medal2016 - clust4_pred)^2, na.rm=TRUE) # Cluster 4 MSE
  clust4_rmse <- sqrt(clust4_mse)

  n4 <- nrow(clust4_test) # Number of records in cluster 4
  
  clust4_dispersion <- summary(clust4_model)$dispersion
  # Cluster 4 Log Likelihood
  clust4_ll <- sum(dnorm(clust4_test$Medal2016, clust4_pred, sqrt(clust4_dispersion), log=TRUE)) 
  
  # Total Predictive Log Likelihood of Task 3 model
  task3_ll <- clust1_ll + clust2_ll + clust3_ll + clust4_ll 
  
  # Weighted MSE & RMSE for all clusters
  task3_mse <- (n1*clust1_mse + n2*clust2_mse + n3*clust3_mse + n4*clust4_mse)/(n1+n2+n3+n4)
  task3_rmse <- (n1*clust1_rmse + n2*clust2_rmse + n3*clust3_rmse + n4*clust4_rmse)/(n1+n2+n3+n4)

  
  # Determining best model based on LL
  if (!is.na(task1_ll) && !is.na(task2_ll) && !is.na(task3_ll)) {
    if(max(task1_ll, task2_ll, task3_ll) == task1_ll) {
      winner_ll[i] <- "Model 1"
    } else if(max(task1_ll, task2_ll, task3_ll) == task2_ll) {
      winner_ll[i] <- "Model 2"
    } else {
      winner_ll[i] <- "Model 3"
    }
  } else {
    next
  }
  
  ### Determine the winner based on MSE
  if(!is.na(task1_mse) & !is.na(task2_mse) & !is.na(task3_mse)) {
    if (min(task1_mse, task2_mse, task3_mse) == task1_mse) {
      winner_mse[i] <- "Model 1"
    } else if (min(task1_mse, task2_mse, task3_mse) == task2_mse) {
      winner_mse[i] <- "Model 2"
    } else {
      winner_mse[i] <- "Model 3"
    }
  } else {
    next
  }
   
 ### Determine the winner based on RMSE
  if(!is.na(task1_rmse) & !is.na(task2_rmse) & !is.na(task3_rmse)) {
    if (min(task1_rmse, task2_rmse, task3_rmse) == task1_rmse) {
      winner_rmse[i] <- "Model 1"
    } else if (min(task1_rmse, task2_rmse, task3_rmse) == task2_rmse) {
      winner_rmse[i] <- "Model 2"
    } else {
      winner_rmse[i] <- "Model 3"
    }
  } else {
    next
  }
}


winner_ll_freq <- table(winner_ll)

barplot(winner_ll_freq, main="Figure 5. Best performing model based on predictive log likelihood", xlab="Model", ylab="Count")

#winner_mse_freq <- table(winner_mse)
#barplot(winner_mse_freq, main="Best performing model based on MSE", xlab="Model", ylab="Count")

winner_rmse_freq <- table(winner_rmse)
barplot(winner_rmse_freq, main="Figure 6. Best performing model based on RMSE", xlab="Model", ylab="Count")



knit("Assessment1v1.R", output = "codeAss.pdf")
