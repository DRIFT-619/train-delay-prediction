# Open this script via the Trains-Delay-Prediction.Rproj file (or set your working
# directory to the project root) so that the relative load("trainsData.RData") below
# resolves correctly.

# Required packages (run once):
# install.packages(c("xgboost", "caret", "Metrics", "ggplot2", "dplyr", "DescTools",
#                     "scales", "ggcorrplot", "GGally", "mgcv", "randomForest"))

library(xgboost)
library(caret)  # For data splitting
library(Metrics)  # For calculating MSE
library(ggplot2)
library(dplyr)
library(DescTools)
library(scales)  # For rescaling
library(ggcorrplot)
library(GGally)
library(mgcv)  # For GAM
library(randomForest)

load("trainsData.RData")

ls()

dim(historicalCongestion)
head(historicalCongestion)

print(historicalCongestion[21,])


dim(trainingData[[1]]$timings)
dim(trainingData[[1]]$congestion)
dim(trainingData[[1]]$arrival)


head(trainingData)

print(trainingData[[1]])

print(testData[[1]])

length(trainingData)

length(testData)

# -----------------------------------------------------------------------------------------

# Extract No. of Entries for Each Day
days <- sapply(trainingData, function(train) train$timings$day.week[1])
table(days)

# Extract train codes and their corresponding days
train_codes <- sapply(trainingData, function(train) train$timings$train.code[1])
train_days <- sapply(trainingData, function(train) train$timings$day.week[1])

# Create a data frame for better visualization
train_schedule <- data.frame(TrainCode = train_codes, Day = train_days)

# Count unique train codes
unique_train_codes <- unique(train_schedule$TrainCode)
num_unique_trains <- length(unique_train_codes)

print(num_unique_trains)

# Check how many times each train code appears on different days
train_occurrences <- table(train_schedule)

# -----------------------------------------------------------------------------------------

# Creating a single Data frame to store important Information using useful Features

textToSeconds <- function(textTime){
  seconds <- as.numeric(strsplit(textTime,split = ":")[[1]]) %*% c(60*60,60,1)
  return(as.numeric(seconds))
}

# Data Preprocessing

# Create a Matrix to store Day and Delay at Sheffield and Nottingham
matrixData <- data.frame(
  week.day = rep(" ", length(trainingData)),
  hour = rep(" ", length(trainingData)),
  
  depDelayLeeds = rep(0, length(trainingData)),
  depDelayWake = rep(0, length(trainingData)),
  depDelayBarn = rep(0, length(trainingData)),
  depDelayMead = rep(0, length(trainingData)),
  
  arrDelayWake = rep(0, length(trainingData)),
  arrDelayBarn = rep(0, length(trainingData)),
  arrDelayMead = rep(0, length(trainingData)),
  arrDelayShef = rep(0, length(trainingData)),
  arrDelayNotts = rep(0, length(trainingData)),
  stringsAsFactors = FALSE
)

# Loop through all the values of Training Data to fill the Matrix Data Variable
for (i in 1:length(trainingData)) {
  dummy <- trainingData[[i]]
  matrixData$week.day[i] <- dummy$timings$day.week[1]
  matrixData$hour[i] <- dummy$congestion$hour[1]
  
  depLeeds <- textToSeconds(tail(dummy$timings$departure.time,4)[1])
  schLeeds <- textToSeconds(tail(dummy$timings$departure.schedule,4)[1])
  matrixData$depDelayLeeds[i] <- depLeeds - schLeeds
  
  depWake <- textToSeconds(tail(dummy$timings$departure.time,3)[1])
  schWake <- textToSeconds(tail(dummy$timings$departure.schedule,3)[1])
  matrixData$depDelayWake[i] <- depWake - schWake
  
  depBarn <- textToSeconds(tail(dummy$timings$departure.time,2)[1])
  schBarn <- textToSeconds(tail(dummy$timings$departure.schedule,2)[1])
  matrixData$depDelayBarn[i] <- depBarn - schBarn
  
  depMead <- textToSeconds(tail(dummy$timings$departure.time,1))
  schMead <- textToSeconds(tail(dummy$timings$departure.schedule,1))
  matrixData$depDelayMead[i] <- depMead - schMead
  
  arrWake <- textToSeconds(tail(dummy$timings$arrival.time,4)[1])
  schWake <- textToSeconds(tail(dummy$timings$arrival.schedule,4)[1])
  matrixData$arrDelayWake[i] <- arrWake - schWake
  
  arrBarn <- textToSeconds(tail(dummy$timings$arrival.time,3)[1])
  schBarn <- textToSeconds(tail(dummy$timings$arrival.schedule,3)[1])
  matrixData$arrDelayBarn[i] <- arrBarn - schBarn
  
  arrMead <- textToSeconds(tail(dummy$timings$arrival.time,2)[1])
  schMead <- textToSeconds(tail(dummy$timings$arrival.schedule,2)[1])
  matrixData$arrDelayMead[i] <- arrMead - schMead
  
  arrSheffield <- textToSeconds(tail(dummy$timings$arrival.time,1))
  schSheffield <- textToSeconds(tail(dummy$timings$arrival.schedule,1))
  matrixData$arrDelayShef[i] <- arrSheffield - schSheffield
  
  matrixData$arrDelayNotts[i] <- dummy$arrival$delay.secs[1]
}

# print(trainingData[[1]])

# head(matrixData)

dim(matrixData)


# Loop through each instance in Training Data
for (i in seq_along(trainingData)) {
  trainingData[[i]]$congestion <- trainingData[[i]]$congestion %>%
    left_join(historicalCongestion, by = c("week.day" = "Day", "hour" = "Hour")) %>%
    mutate(
      Leeds_train_diff = (trainingData[[i]]$congestion$Leeds.trains - Leeds.trains.y) / Leeds.trains.y,
      Sheff_train_diff = (trainingData[[i]]$congestion$Sheffield.trains - Sheffield.trains.y) / Sheffield.trains.y,
      Notts_train_diff = (trainingData[[i]]$congestion$Nottingham.trains - Nottingham.trains.y) / Nottingham.trains.y,
      
      Leeds_delay_diff = (trainingData[[i]]$congestion$Leeds.av.delay - Leeds.av.delay.y), # / Leeds.av.delay.y,
      Sheff_delay_diff = (trainingData[[i]]$congestion$Sheffield.av.delay - Sheffield.av.delay.y), # / Sheffield.av.delay.y,
      Notts_delay_diff = (trainingData[[i]]$congestion$Nottingham.av.delay - Nottingham.av.delay.y), # / Nottingham.av.delay.y,
      
      high_congestion = ifelse(trainingData[[i]]$congestion$Leeds.trains > Leeds.trains.y, 1, 0),
      high_delay = ifelse(trainingData[[i]]$congestion$Leeds.av.delay > Leeds.av.delay.y + 2, 1, 0)  # 2 sec Threshold
    ) %>%
    select(-ends_with(".y"))  # Remove duplicate columns after join
}


# Initialize empty dataframe to store extracted features
congestion_features <- data.frame()

# Loop through trainingData to extract features
for (i in seq_along(trainingData)) {
  if (!is.null(trainingData[[i]]$congestion)) {
    temp_data <- trainingData[[i]]$congestion %>%
      select(week.day, hour, Leeds_train_diff, Sheff_train_diff, Notts_train_diff, 
             Leeds_delay_diff, Sheff_delay_diff, Notts_delay_diff, high_congestion, high_delay)
    
    # Append extracted features to congestion_features
    congestion_features <- bind_rows(congestion_features, temp_data)
  }
}


# Ensure `hour` is an integer in both datasets
matrixData <- matrixData %>%
  mutate(hour = as.integer(hour))

congestion_features <- congestion_features %>%
  mutate(hour = as.integer(hour))

matrixData$hour_sin <- sin(2 * pi * matrixData$hour / 24)
matrixData$hour_cos <- cos(2 * pi * matrixData$hour / 24)


if (nrow(matrixData) == nrow(congestion_features)) {
  # Append all congestion features directly to matrixData
  matrixData <- cbind(matrixData, congestion_features)
  
  cat("Congestion features successfully added to matrixData! \n")
} else {
  cat("Error: Row count mismatch between matrixData and congestion_features \n")
}

# Encode day as a binary weekday/weekend flag (3-group and 0-6 numeric encodings were
# also tried and dropped in favour of this simpler split)
matrixData$day_encoded <- ifelse(matrixData$week.day %in% c("Saturday", "Sunday"), 1, 0)

# head(matrixData)

dim(matrixData)


# Remove Redundant Features
matrixData <- matrixData %>%
  select(-c(week.day, hour))

# head(matrixData)

colnames(matrixData)

dim(matrixData)

# -----------------------------------------------------------------------------------------

# Checking Correlation between Departure Features and Target Variable

# Compute Normal Correlation
for (col_name in colnames(matrixData)) {
  if (col_name != "arrDelayNotts") {
    correlation <- cor(matrixData[[col_name]], matrixData$arrDelayNotts, use="complete.obs")
    cat("Correlation of", col_name, ":", correlation, "\n")
  }
}


# Determine Redundant Features by checking Correlation with other features in the Dataset

# Compute correlation matrix for numeric features in matrixData
cor_matrix <- cor(matrixData[, sapply(matrixData, is.numeric)], use = "complete.obs")

# Print correlation matrix
print(cor_matrix)

# Identify highly correlated features (above 0.8)
high_corr_pairs <- which(abs(cor_matrix) > 0.8 & abs(cor_matrix) < 1, arr.ind = TRUE)

# Print feature pairs with high correlation
if (length(high_corr_pairs) > 0) {
  for (i in 1:nrow(high_corr_pairs)) {
    cat("High correlation between:", colnames(matrixData)[high_corr_pairs[i, 1]], 
        "and", colnames(matrixData)[high_corr_pairs[i, 2]], "\n")
  }
} else {
  cat("No strongly correlated features found! \n")
}

# -----------------------------------------------------------------------------------------

# Graphs & Plots ->

# Plots for Numerical Features ->

# Scatter plot of each candidate predictor against the target (arrDelayNotts).
# Originally written as 15 near-identical copy-pasted ggplot blocks; collapsed into
# a loop here since the plots are independent, view-only, and don't feed downstream code.
scatter_vs_target <- function(df, x_col, x_label) {
  ggplot(df, aes(x = .data[[x_col]], y = arrDelayNotts)) +
    geom_point(alpha = 0.5, color = "blue") +
    geom_smooth(method = "lm", color = "red") +
    labs(title = paste0("Scatter Plot: ", x_label, " vs Nottingham Delay"),
         x = x_label,
         y = "Nottingham Delay (seconds)") +
    theme_minimal()
}

scatter_features <- c(
  depDelayLeeds = "Leeds Departure Delay", depDelayWake = "Wake Departure Delay",
  depDelayBarn = "Barnsley Departure Delay", depDelayMead = "Meadowhall Departure Delay",
  arrDelayWake = "Wake Arrival Delay", arrDelayBarn = "Barnsley Arrival Delay",
  arrDelayMead = "Meadowhall Arrival Delay", arrDelayShef = "Sheffield Arrival Delay",
  hour_sin = "Hour (sin)", hour_cos = "Hour (cos)",
  Leeds_train_diff = "Leeds Train Congestion Diff", Sheff_train_diff = "Sheffield Train Congestion Diff",
  Notts_train_diff = "Nottingham Train Congestion Diff", Leeds_delay_diff = "Leeds Avg Delay Diff",
  Sheff_delay_diff = "Sheffield Avg Delay Diff", Notts_delay_diff = "Nottingham Avg Delay Diff"
)

for (col in names(scatter_features)) {
  print(scatter_vs_target(matrixData, col, scatter_features[[col]]))
}


# Plots for Categorical Features

# Box Plot
ggplot(matrixData, aes(x = factor(high_congestion), y = arrDelayNotts)) +
  geom_boxplot(fill = "blue", alpha = 0.5) +
  theme_minimal() +
  labs(title = "Box Plot of Target Variable by Categorical Feature",
       x = "High_Congestion",
       y = "Nottingham Delays")

# Box Plot
ggplot(matrixData, aes(x = factor(high_delay), y = arrDelayNotts)) +
  geom_boxplot(fill = "blue", alpha = 0.5) + # , outlier.shape = NA) +
  theme_minimal() +
  # coord_cartesian(ylim = c(0, 5000)) +  # Adjust these limits as needed
  labs(title = "Box Plot of Target Variable by Categorical Feature",
       x = "High_Delay",
       y = "Nottingham Delays")

# Box Plot
ggplot(matrixData, aes(x = factor(day_encoded), y = arrDelayNotts)) +
  geom_boxplot(fill = "blue", alpha = 0.5) +
  theme_minimal() +
  labs(title = "Box Plot of Target Variable by Categorical Feature",
       x = "Day_Encoded",
       y = "Nottingham Delays")



# Distribution of Arrival Delay at Nottingham
ggplot(matrixData, aes(x = arrDelayNotts)) +
geom_histogram(binwidth = 50, fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Nottingham Delay",
       x = "Nottingham Delay (seconds)",
       y = "Frequency") +
  theme_minimal()


# Arrival Delay at Nottingham
plot(matrixData$arrDelayNotts, type="l", col="blue", main="Arrival Delay at Nottingham")

# Compute mean & standard deviation of delay for each day
day_delay_stats <- matrixData %>%
  group_by(day) %>%
  summarise(
    mean_delay = mean(arrDelayNotts, na.rm = TRUE),
    sd_delay = sd(arrDelayNotts, na.rm = TRUE),
    count = n()
  )

# Print delay stats for each day
print(day_delay_stats)


# Boxplot to visualize delay distribution for each day
ggplot(matrixData, aes(x = day, y = arrDelayNotts, fill = day)) +
  geom_boxplot() +
  labs(title = "Distribution of Arrival Delay at Nottingham by Day",
       x = "Day of the Week",
       y = "Delay at Nottingham (seconds)") +
  theme_minimal()


# Box Plot for Munerical Features
matrixData$Sheff_bin <- cut(matrixData$arrDelayShef, breaks = 5)  # Creates 5 bins

ggplot(matrixData, aes(x = Sheff_bin, y = arrDelayNotts)) +
  geom_boxplot(fill = "blue", alpha = 0.5, outlier.colour = "red", outlier.shape = 16) +
  theme_minimal() +
  labs(title = "Box Plot of Target Variable by Binned arrDelaySheff",
       x = "Binned arrDelaySheff",
       y = "arrDelayNotts") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotates x-axis labels



# Calculate correlation matrix
cor_matrix <- cor(matrixData[, sapply(matrixData, is.numeric)], use = "complete.obs")

# Plot the heatmap
ggcorrplot(cor_matrix, method = "circle", type = "lower", lab = TRUE)


# Select the numerical features you want to plot

selected_features <- matrixData[, c("arrDelayShef", "arrDelayWake", "arrDelayBarn", "arrDelayMead", "arrDelayNotts")]

# Create the pair plot
ggpairs(selected_features, 
        title = "Pair Plot of Selected Features",
        progress = FALSE)

# ---------------------------------------------------------------------------------------

# Feature Selection ->

colnames(matrixData)

# Drop the binary congestion/delay flags and day encoding — these were explored during
# EDA (see box plots) but didn't add predictive value over the continuous versions
matrixData <- matrixData[, !colnames(matrixData) %in% c("high_congestion", "high_delay", "day_encoded")]

dim(matrixData)



# 1. Departure Delay Leeds -> [17]
matrixData$abs_col_1 <- abs(matrixData$depDelayLeeds)

matrixData$col_1_sq <- matrixData$depDelayLeeds^2
matrixData$col_1_cu <- matrixData$depDelayLeeds^3

matrixData$col_1_Sub_1 <- matrixData$depDelayLeeds - matrixData$depDelayWake
matrixData$col_1_Sub_2 <- matrixData$depDelayLeeds - matrixData$depDelayBarn
matrixData$col_1_Sub_3 <- matrixData$depDelayLeeds - matrixData$depDelayMead
matrixData$col_1_Sub_4 <- matrixData$depDelayLeeds - matrixData$arrDelayWake
matrixData$col_1_Sub_5 <- matrixData$depDelayLeeds - matrixData$arrDelayBarn
matrixData$col_1_Sub_6 <- matrixData$depDelayLeeds - matrixData$arrDelayMead
matrixData$col_1_Sub_7 <- matrixData$depDelayLeeds - matrixData$arrDelayShef

matrixData$col_1_abs_1 <- abs(matrixData$col_1_Sub_1)
matrixData$col_1_abs_2 <- abs(matrixData$col_1_Sub_2)
matrixData$col_1_abs_3 <- abs(matrixData$col_1_Sub_3)
matrixData$col_1_abs_4 <- abs(matrixData$col_1_Sub_4)
matrixData$col_1_abs_5 <- abs(matrixData$col_1_Sub_5)
matrixData$col_1_abs_6 <- abs(matrixData$col_1_Sub_6)
matrixData$col_1_abs_7 <- abs(matrixData$col_1_Sub_7)


# 2. Departure Delay Wake -> [15]
matrixData$abs_col_2 <- abs(matrixData$depDelayWake)

matrixData$col_2_sq <- matrixData$depDelayWake^2
matrixData$col_2_cu <- matrixData$depDelayWake^3

matrixData$col_2_Sub_1 <- matrixData$depDelayWake - matrixData$depDelayBarn
matrixData$col_2_Sub_2 <- matrixData$depDelayWake - matrixData$depDelayMead
matrixData$col_2_Sub_3 <- matrixData$depDelayWake - matrixData$arrDelayWake
matrixData$col_2_Sub_4 <- matrixData$depDelayWake - matrixData$arrDelayBarn
matrixData$col_2_Sub_5 <- matrixData$depDelayWake - matrixData$arrDelayMead
matrixData$col_2_Sub_6 <- matrixData$depDelayWake - matrixData$arrDelayShef

matrixData$col_2_abs_1 <- abs(matrixData$col_2_Sub_1)
matrixData$col_2_abs_2 <- abs(matrixData$col_2_Sub_2)
matrixData$col_2_abs_3 <- abs(matrixData$col_2_Sub_3)
matrixData$col_2_abs_4 <- abs(matrixData$col_2_Sub_4)
matrixData$col_2_abs_5 <- abs(matrixData$col_2_Sub_5)
matrixData$col_2_abs_6 <- abs(matrixData$col_2_Sub_6)


# 3. Departure Delay Barn -> [13]
matrixData$abs_col_3 <- abs(matrixData$depDelayBarn)

matrixData$col_3_sq <- matrixData$depDelayBarn^2
matrixData$col_3_cu <- matrixData$depDelayBarn^3

matrixData$col_3_Sub_1 <- matrixData$depDelayBarn - matrixData$depDelayMead
matrixData$col_3_Sub_2 <- matrixData$depDelayBarn - matrixData$arrDelayWake
matrixData$col_3_Sub_3 <- matrixData$depDelayBarn - matrixData$arrDelayBarn
matrixData$col_3_Sub_4 <- matrixData$depDelayBarn - matrixData$arrDelayMead
matrixData$col_3_Sub_5 <- matrixData$depDelayBarn - matrixData$arrDelayShef

matrixData$col_3_abs_1 <- abs(matrixData$col_3_Sub_1)
matrixData$col_3_abs_2 <- abs(matrixData$col_3_Sub_2)
matrixData$col_3_abs_3 <- abs(matrixData$col_3_Sub_3)
matrixData$col_3_abs_4 <- abs(matrixData$col_3_Sub_4)
matrixData$col_3_abs_5 <- abs(matrixData$col_3_Sub_5)


# 4. Departure Delay Mead -> [11]
matrixData$abs_col_4 <- abs(matrixData$depDelayMead)

matrixData$col_4_sq <- matrixData$depDelayMead^2
matrixData$col_4_cu <- matrixData$depDelayMead^3

matrixData$col_4_Sub_1 <- matrixData$depDelayMead - matrixData$arrDelayWake
matrixData$col_4_Sub_2 <- matrixData$depDelayMead - matrixData$arrDelayBarn
matrixData$col_4_Sub_3 <- matrixData$depDelayMead - matrixData$arrDelayMead
matrixData$col_4_Sub_4 <- matrixData$depDelayMead - matrixData$arrDelayShef

matrixData$col_4_abs_1 <- abs(matrixData$col_4_Sub_1)
matrixData$col_4_abs_2 <- abs(matrixData$col_4_Sub_2)
matrixData$col_4_abs_3 <- abs(matrixData$col_4_Sub_3)
matrixData$col_4_abs_4 <- abs(matrixData$col_4_Sub_4)


# 5. Arrival Delay Wake -> [9]
matrixData$abs_col_5 <- abs(matrixData$arrDelayWake)

matrixData$col_5_sq <- matrixData$arrDelayWake^2
matrixData$col_5_cu <- matrixData$arrDelayWake^3

matrixData$col_5_Sub_1 <- matrixData$arrDelayWake - matrixData$arrDelayBarn
matrixData$col_5_Sub_2 <- matrixData$arrDelayWake - matrixData$arrDelayMead
matrixData$col_5_Sub_3 <- matrixData$arrDelayWake - matrixData$arrDelayShef

matrixData$col_5_abs_1 <- abs(matrixData$col_5_Sub_1)
matrixData$col_5_abs_2 <- abs(matrixData$col_5_Sub_2)
matrixData$col_5_abs_3 <- abs(matrixData$col_5_Sub_3)


# 6. Arrival Delay Barn -> [7]
matrixData$abs_col_6 <- abs(matrixData$arrDelayBarn)

matrixData$col_6_sq <- matrixData$arrDelayBarn^2
matrixData$col_6_cu <- matrixData$arrDelayBarn^3

matrixData$col_6_Sub_1 <- matrixData$arrDelayBarn - matrixData$arrDelayMead
matrixData$col_6_Sub_2 <- matrixData$arrDelayBarn - matrixData$arrDelayShef

matrixData$col_6_abs_1 <- abs(matrixData$col_5_Sub_1)
matrixData$col_6_abs_2 <- abs(matrixData$col_5_Sub_2)


# 7. Arrival Delay Mead -> [5]
matrixData$abs_col_7 <- abs(matrixData$arrDelayMead)

matrixData$col_7_sq <- matrixData$arrDelayMead^2
matrixData$col_7_cu <- matrixData$arrDelayMead^3

matrixData$col_7_Sub_1 <- matrixData$arrDelayMead - matrixData$arrDelayShef

matrixData$col_7_abs_1 <- abs(matrixData$col_5_Sub_1)


# 8. Arrival Delay Shef -> [3]
matrixData$abs_col_8 <- abs(matrixData$arrDelayShef)

matrixData$col_8_sq <- matrixData$arrDelayShef^2
matrixData$col_8_cu <- matrixData$arrDelayShef^3


# Hour in Sin & Cos -> [7]
matrixData$hour_sin_cos <- matrixData$hour_sin * matrixData$hour_cos

matrixData$hour_sin_sq <- matrixData$hour_sin^2
matrixData$hour_sin_cu <- matrixData$hour_sin^3

matrixData$hour_cos_sq <- matrixData$hour_cos^2
matrixData$hour_cos_cu <- matrixData$hour_cos^3

matrixData$hour_sin_cos_sq <- matrixData$hour_sin_cos^2
matrixData$hour_sin_cos_cu <- matrixData$hour_sin_cos^3


# 9. Train Difference -> [12]
matrixData$train_leeds_sq <- matrixData$Leeds_train_diff^2
matrixData$train_leeds_cu <- matrixData$Leeds_train_diff^3

matrixData$train_sheff_sq <- matrixData$Sheff_train_diff^2
matrixData$train_sheff_cu <- matrixData$Sheff_train_diff^3

matrixData$train_notts_sq <- matrixData$Notts_train_diff^2
matrixData$train_notts_cu <- matrixData$Notts_train_diff^3

matrixData$col_9_Sub_1 <- matrixData$Leeds_train_diff - matrixData$Sheff_train_diff
matrixData$col_9_Sub_2 <- matrixData$Leeds_train_diff - matrixData$Notts_train_diff
matrixData$col_9_Sub_3 <- matrixData$Sheff_train_diff - matrixData$Notts_train_diff

matrixData$col_9_abs_1 <- abs(matrixData$col_9_Sub_1)
matrixData$col_9_abs_2 <- abs(matrixData$col_9_Sub_2)
matrixData$col_9_abs_3 <- abs(matrixData$col_9_Sub_3)


# 10. Delay Difference -> [12]
matrixData$delay_leeds_sq <- matrixData$Leeds_delay_diff^2
matrixData$delay_leeds_cu <- matrixData$Leeds_delay_diff^3

matrixData$delay_sheff_sq <- matrixData$Sheff_delay_diff^2
matrixData$delay_sheff_cu <- matrixData$Sheff_delay_diff^3

matrixData$delay_notts_sq <- matrixData$Notts_delay_diff^2
matrixData$delay_notts_cu <- matrixData$Notts_delay_diff^3

matrixData$col_10_Sub_1 <- matrixData$Leeds_delay_diff - matrixData$Sheff_delay_diff
matrixData$col_10_Sub_2 <- matrixData$Leeds_delay_diff - matrixData$Notts_delay_diff
matrixData$col_10_Sub_3 <- matrixData$Sheff_delay_diff - matrixData$Notts_delay_diff

matrixData$col_10_abs_1 <- abs(matrixData$col_10_Sub_1)
matrixData$col_10_abs_2 <- abs(matrixData$col_10_Sub_2)
matrixData$col_10_abs_3 <- abs(matrixData$col_10_Sub_3)


# Train & Delay Interaction -> [6] Total = 117
matrixData$train_leeds_interaction <- matrixData$Leeds_train_diff * matrixData$depDelayLeeds
matrixData$train_sheff_interaction <- matrixData$Sheff_train_diff * matrixData$arrDelayShef
matrixData$train_notts_interaction <- matrixData$Notts_train_diff * matrixData$arrDelayShef

matrixData$delay_leeds_interaction <- matrixData$Leeds_delay_diff * matrixData$depDelayLeeds
matrixData$delay_sheff_interaction <- matrixData$Sheff_delay_diff * matrixData$arrDelayShef
matrixData$delay_notts_interaction <- matrixData$Notts_delay_diff * matrixData$arrDelayShef



cat("Interaction features successfully added to matrixData! \n")

dim(matrixData)

# -------------------------------------------------------------------------------------

# Check Highly Correlated Features ->

# Function to Remove Redundant Features (Retain one of each correlated pair)
find_redundant_features <- function(data, threshold = 0.8) {
  # Calculate Correlation Matrix
  corr_matrix <- cor(data, use = "complete.obs")
  
  # Find Highly Correlated Feature Pairs
  high_corr_pairs <- which(abs(corr_matrix) > threshold, arr.ind = TRUE)
  high_corr_pairs <- high_corr_pairs[high_corr_pairs[, 1] < high_corr_pairs[, 2], ]  # Keep only upper triangle
  
  # Store redundant features to drop
  redundant_features <- c()
  retained_features <- c()
  
  for (i in 1:nrow(high_corr_pairs)) {
    feature_1 <- colnames(corr_matrix)[high_corr_pairs[i, 1]]
    feature_2 <- colnames(corr_matrix)[high_corr_pairs[i, 2]]
    
    # Check if either feature is already marked as redundant
    if (feature_1 %in% redundant_features || feature_2 %in% redundant_features) {
      next  # Skip if already marked for removal
    }
    
    # If not, retain one feature and mark the other as redundant
    redundant_features <- c(redundant_features, feature_2)
  }
  
  return(unique(redundant_features))
}

# Exclude the target variable from redundancy check
filtered_data <- matrixData[, colnames(matrixData) != "label"]

# Find Redundant Features
redundant_features <- find_redundant_features(filtered_data, threshold = 0.8)
cat("Redundant Features Detected:\n")
print(redundant_features)

print(length(redundant_features))

# Drop Redundant Features from matrixData
matrixData <- matrixData[, !colnames(matrixData) %in% redundant_features]
cat("Redundant Features Dropped Successfully!\n")


colnames(matrixData)

# Show the new dimensions of the data frame
dim(matrixData)





# -------------------------------------------------------------------------------------

# Scaling & Log Transformation ->

# Log Transformation for Target Variable (Optional but Recommended)
matrixData$arrDelayNotts <- log(matrixData$arrDelayNotts + abs(min(matrixData$arrDelayNotts)) + 1)

# Store Original Data
original_matrixData <- matrixData

# Feature Scaling (Excluding Target Variable)
scale_features <- function(df, target_col) {
  df_scaled <- df
  
  for (col in colnames(df)) {
    if (is.numeric(df[[col]]) && col != target_col) { 
      # Min-Max Scaling (0 to 1)
      min_val <- min(df[[col]], na.rm = TRUE)
      max_val <- max(df[[col]], na.rm = TRUE)
      
      if (min_val != max_val) {  # Avoid division by zero
        df_scaled[[col]] <- rescale(df[[col]], to = c(0, 1))
      }
    }
  }
  return(df_scaled)
}

# Apply Feature Scaling (Excluding arrDelayNotts)
matrixData <- scale_features(matrixData, target_col = "arrDelayNotts")

# View the Scaled Data
# head(matrixData)

dim(matrixData)

# -------------------------------------------------------------------------------------

# Outlier Removal ->
# A Z-score based approach (threshold = 4 SD) was tried first but the IQR method below
# gave better held-out performance, so that's what's used.

# Check missing values in each column
colSums(is.na(matrixData))

matrixDataOriginal <- matrixData

cat("Before Outlier Removal: Min =", min(matrixData$arrDelayNotts), "Max =", max(matrixData$arrDelayNotts), "\n")

# IQR Method ->
remove_outliers_iqr <- function(df, target_col) {
  for (col in colnames(df)) {
    if (is.numeric(df[[col]]) && col != target_col) { 
      Q1 <- quantile(df[[col]], 0.25, na.rm = TRUE)
      Q3 <- quantile(df[[col]], 0.75, na.rm = TRUE)
      IQR <- Q3 - Q1
      lower_bound <- Q1 - 2 * IQR # 3, 2.5, 2
      upper_bound <- Q3 + 2 * IQR # 3, 2.5, 2 
      df <- df[df[[col]] >= lower_bound & df[[col]] <= upper_bound, ]
    }
  }
  return(df)
}

matrixData <- remove_outliers_iqr(matrixData, target_col = "arrDelayNotts")

cat("After Outlier Removal: Min =", min(matrixData$arrDelayNotts), "Max =", max(matrixData$arrDelayNotts), "\n")

dim(matrixDataOriginal)

dim(matrixData)


# Check how many rows are left after filtering
cat("Rows before filtering:", nrow(matrixData), "\n")

# Filter rows where the target variable (arrDelayNotts) is <= 600
matrixData <- matrixData[matrixData$arrDelayNotts <= 600, ]

cat("Rows after filtering:", nrow(matrixData), "\n")
cat("After Outlier Removal: Min =", min(matrixData$arrDelayNotts), "Max =", max(matrixData$arrDelayNotts), "\n")


# Add 1000 to all values of the target variable
# matrixData$arrDelayNotts <- matrixData$arrDelayNotts + 1000

# --------------------------------------------------------------------------------------

# Imputing Missing Values

# Check missing values in each column
colSums(is.na(matrixData))

# Step 1: Compute and Store Mean for All Columns (Even If No Missing Values)
mean_values <- list()  # Dictionary to store column-wise means

for (col in colnames(matrixData)) {
  mean_values[[col]] <- mean(matrixData[[col]], na.rm = TRUE)  # Store mean of each column
}

cat("Stored mean values for all columns! \n")

# Step 2: Impute Missing Values in Training Data Using Stored Means
for (col in colnames(matrixData)) {
  if (any(is.na(matrixData[[col]]))) {  # Only impute if missing values exist
    matrixData[[col]][is.na(matrixData[[col]])] <- mean_values[[col]]
  }
}

# Check missing values in each column
colSums(is.na(matrixData))

cat("Missing values Imputed with column mean! \n")

# --------------------------------------------------------------------------------------

# PCA — EXPLORED, NOT USED IN FINAL MODEL
# 20 principal components explained ~88% of variance, but training on them did not
# beat training on the original named/engineered features, so this path was abandoned.
# Left here (disabled) for the record — do NOT uncomment, since it overwrites matrixData
# with PC columns, which breaks the named `selected_features` list used below.

# features_df <- data.frame(arrDelayNotts = matrixData$arrDelayNotts)
# matrixData_no_target <- matrixData[, !colnames(matrixData) %in% c("arrDelayNotts")]
# matrix_pca <- prcomp(matrixData_no_target, center = TRUE, scale. = TRUE)
# matrix_pca_df <- as.data.frame(matrix_pca$x[, 1:20])
# colnames(matrix_pca_df) <- paste0("pc", 1:20)
# explained_variance <- summary(matrix_pca)$importance[2, 1:20] * 100
# cat("Total Variance Explained:", sum(explained_variance), "%\n")  # ~88%
# matrixData_pca <- cbind(matrix_pca_df, features_df)

# --------------------------------------------------------------------------------------

# XGBoost

set.seed(42)  # For reproducibility

# Define features (X) and target variable (y)
X <- matrixData[, colnames(matrixData) != "arrDelayNotts"]  # Exclude Target Variable
X <- data.matrix(X)
y <- matrixData$arrDelayNotts  # Target Variable

sum(is.na(y))   # Counts NA values
sum(is.nan(y))  # Counts NaN values

# y[is.na(y)] <- mean(y, na.rm = TRUE)

# Split data: 80% train, 20% test
train_indices <- createDataPartition(y, p = 0.8, list = FALSE)

X_train <- X[train_indices, , drop = FALSE]
y_train <- y[train_indices]
X_test <- X[-train_indices, , drop = FALSE]
y_test <- y[-train_indices]

dim(X_train)

cat("Train Target Range:", min(y_train), "to", max(y_train), "\n")
cat("Test Target Range:", min(y_test), "to", max(y_test), "\n")

# Convert to DMatrix for XGBoost
train_matrix <- xgb.DMatrix(data = X_train, label = y_train)
test_matrix <- xgb.DMatrix(data = X_test, label = y_test)


# Define XGBoost parameters
params <- list(
  objective = "reg:squarederror",  # Regression task
  eval_metric = "rmse",  # Root Mean Squared Error
  max_depth = 3,  
  eta = 0.1  
)

# Perform 5-fold Cross-Validation (CV)
cv_results <- xgb.cv(
  params = params,
  data = train_matrix,
  nfold = 5,
  nrounds = 100,
  early_stopping_rounds = 10,  # Stops training if test rmse doesn't improve for 10 rounds
  verbose = 0
)

# Print best iteration from CV
best_nrounds <- cv_results$best_iteration
cat("Best iteration from CV:", best_nrounds, "\n")



# Recursive Feature Elimination (RFE) using caret's rfe() with an xgbTree model was run
# separately to search subset sizes c(5,7,10,15,20,25,30,35,40,45,50,55,60) via 5-fold CV.
# A 50-feature subset and a 25-feature subset were both evaluated (see Images/); the
# 50-feature subset below gave the best held-out MSE and is what's used in the final model.
# The RFE run itself is omitted here since it takes a while — this is its output, hardcoded:
selected_features <- c("col_8_sq", "depDelayMead", "col_2_Sub_6", "col_2_Sub_4", "delay_sheff_interaction", "col_4_Sub_4", "depDelayWake", "col_1_Sub_2", "Leeds_delay_diff", "col_1_Sub_3", "train_notts_interaction", "train_sheff_sq", "col_4_abs_3", "col_9_Sub_1", "col_2_Sub_3", "col_1_Sub_1", "col_3_Sub_1", "col_3_Sub_3", "col_9_Sub_3", "hour_cos", "delay_sheff_sq", "hour_sin", "train_leeds_sq", "depDelayLeeds", "col_10_abs_1", "col_4_Sub_3", "Sheff_delay_diff", "Leeds_train_diff", "Notts_delay_diff", "hour_sin_cos", "train_notts_sq", "col_2_sq", "col_9_abs_2", "col_2_Sub_5", "col_1_abs_1", "col_9_Sub_2", "col_9_abs_3", "Notts_train_diff", "col_1_abs_2", "col_1_abs_3", "col_1_cu", "col_10_abs_2", "col_10_abs_3", "col_10_Sub_1", "col_2_abs_3", "col_2_Sub_1", "col_3_abs_1", "col_3_abs_3", "col_4_abs_1", "col_9_abs_1")

# Create Train & Test datasets with only selected features
X_train <- X_train[, selected_features, drop = FALSE]
X_test <- X_test[, selected_features, drop = FALSE]

dim(X_train)

# Hyperparameter Tuning ->
# Grid search (nrounds, max_depth, eta, gamma, colsample_bytree, min_child_weight,
# subsample) was run via caret::train() with 5-fold CV — omitted here for runtime,
# its best result (max_depth = 3, eta = 0.1) is hardcoded into the model below.

# Build XGBoost Model
xgb_model <- xgboost(
  data = X_train,
  label = y_train,
  nrounds = best_nrounds, # 100  # Use the best iteration found in CV
  max_depth = 3,
  eta = 0.1,  
  objective = "reg:squarederror",
  eval_metric = "mae", # "rmse",
  early_stopping_rounds = 10,  # Stops training if test rmse doesn't improve for 10 rounds
  verbose = 0
)

# Predict on test data
y_pred <- predict(xgb_model, X_test)

cat("Predicted Target Range:", min(y_pred), "to", max(y_pred), "\n")

# Model Evaluation ->

# Calculate Mean Squared Error (MSE)
mse_value <- mse(y_test, y_pred)
cat("Mean Squared Error (MSE):", mse_value, "\n")

# Calculate Mean Absolute Error (MAE)
mae_value <- mae(y_test, y_pred)
cat("Mean Absolute Error (MAE):", mae_value, "\n")

# Feature Importance Plot
importance_matrix <- xgb.importance(model = xgb_model)
xgb.plot.importance(importance_matrix, top_n = 15)

# Print feature importance values
# print(importance_matrix)

print(importance_matrix$Feature[1:60])



# Compute MAPE for IQR & Z-Score ->
mape <- function(actual, predicted) {
  mean(abs((actual - predicted) / actual), na.rm = TRUE) * 100
}

mape_score <- mape(y_test, y_pred)

cat("MAPE Score:", mape_score, "%\n")



# Comparing Training and Test MSE & MAE ->
train_pred <- predict(xgb_model, X_train)

mse_train <- mse(y_train, train_pred)
mae_train <- mae(y_train, train_pred)

cat("Train MSE:", mse_train, "Test MSE:", mse_value, "\n")
cat("Train MAE:", mae_train, "Test MAE:", mae_value, "\n")

# -------------------------------------------------------------------------------------

# Training GAM Model ->

# Step 1: Prepare Data for GAM

# Convert Data to DataFrame for GAM
X_train_gam <- as.data.frame(X_train)
X_test_gam <- as.data.frame(X_test)

dim(X_train_gam)

# Add Target Variable for GAM Training
train_data_gam <- cbind(X_train_gam, arrDelayNotts = y_train)
test_data_gam <- X_test_gam

dim(train_data_gam)

# Define Formula for GAM
gam_formula <- as.formula(paste("arrDelayNotts ~", paste(paste0("s(", selected_features, ", k = 10)"), collapse = " + ")))

# Step 2: Hyperparameter Tuning for GAM
set.seed(42)

# Grid search over method = {REML, GCV.Cp} x select = {TRUE, FALSE} was run, evaluating
# each combination's held-out MSE — omitted here for runtime, best result hardcoded below.
best_params <- list(method = "REML", select = FALSE)

best_gam_model <- gam(gam_formula, data = train_data_gam, method = best_params$method, select = best_params$select)

# Use Best Model for Predictions
y_pred_gam <- predict(best_gam_model, newdata = test_data_gam)

# Calculate MAE
mse_value_gam <- mse(y_test, y_pred_gam)
mae_value_gam <- mae(y_test, y_pred_gam)
cat("GAM Model Performance -> MSE:", mse_value_gam, "MAE:", mae_value_gam, "\n")

# -------------------------------------------------------------------------------------

# Model Blending (XGBoost + GAM) ->

# Combine predictions using a weighted average
weight_xgb <- 0.2
weight_gam <- 0.8

y_pred_blended <- (weight_xgb * y_pred) + (weight_gam * y_pred_gam)

# Evaluate Blended Model
mse_value_blended <- mse(y_test, y_pred_blended)
mae_value_blended <- mae(y_test, y_pred_blended)

cat("\nBlended Model - Mean Squared Error (MSE):", mse_value_blended, "\n")
cat("Blended Model - Mean Absolute Error (MAE):", mae_value_blended, "\n")




# Blending Function
blend_models <- function(y_pred_xgb, y_pred_gam, weight_xgb) {
  weight_gam <- 1 - weight_xgb
  blended_pred <- (weight_xgb * y_pred_xgb) + (weight_gam * y_pred_gam)
  return(blended_pred)
}

# Generate predictions on validation set (X_test)
y_pred_xgb <- predict(xgb_model, X_test)
y_pred_gam <- predict(best_gam_model, newdata = test_data_gam)

# Define a range of weights to try
weights <- seq(0, 1, by = 0.05)  # Adjust step size for finer tuning

# Store results
results <- data.frame(Weight_XGB = numeric(), MSE = numeric(), MAE = numeric())

# Try different weights
for (weight_xgb in weights) {
  blended_pred <- blend_models(y_pred_xgb, y_pred_gam, weight_xgb)
  
  # Evaluate Performance
  mse_value <- mse(y_test, blended_pred)
  mae_value <- mae(y_test, blended_pred)
  
  # Save results
  results <- rbind(results, data.frame(Weight_XGB = weight_xgb, MSE = mse_value, MAE = mae_value))
}

# Display Results
print(results)

# Plot Results to Visualize Best Weight
library(ggplot2)
ggplot(results, aes(x = Weight_XGB, y = MSE)) +
  geom_line(color = "blue") +
  geom_point(color = "red") +
  labs(title = "Blending Weight Optimization (XGBoost vs. GAM)",
       x = "XGBoost Weight",
       y = "MSE") +
  theme_minimal()

# Get Best Weight Based on Minimum MSE
best_weight_row <- results[which.min(results$MSE), ]
best_weight_xgb <- best_weight_row$Weight_XGB
best_weight_gam <- 1 - best_weight_xgb

cat("Best Weight for XGBoost:", best_weight_xgb, "\n")
cat("Best Weight for GAM:", best_weight_gam, "\n")
cat("Corresponding MSE:", best_weight_row$MSE, "and MAE:", best_weight_row$MAE, "\n")

# -------------------------------------------------------------------------------------

# Linear Regression

set.seed(42)  # Ensure reproducibility

# Define features (X) and target variable (y)

# X <- matrixData[, colnames(matrixData) != "arrDelayNotts"]
# X <- X[, colnames(X) != "day"] # Dropping Days column
X <- matrixData$arrDelayShef
y <- matrixData$arrDelayNotts  # Target: Nottingham delay

# Split data: 80% train, 20% test
train_indices <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_indices]
y_train <- y[train_indices]

X_test <- X[-train_indices]
y_test <- y[-train_indices]

# Train model
lm_model <- lm(y_train ~ X_train)

# Print summary of the model
summary(lm_model)

# Predict on test data
y_pred <- predict(lm_model, data.frame(X_train = X_test))

# Calculate Mean Squared Error (MSE)
mse_value <- mse(y_test, y_pred)

cat("Mean Squared Error (MSE) for Linear Regression:", mse_value, "\n")

# Calculate Mean Squared Error (MSE)
mae_value <- mae(y_test, y_pred)

cat("Mean Absolute Error (MAE) for Linear Regression:", mae_value, "\n")

# ----------------------------------------------------------------------------------------

# Make Predictions on testData [Not X_test] before saving it to the CSV File

# Step 1 -> Data Preprocessing

# Create a Matrix to store Day and Delay at Sheffield and Nottingham
testMatrixData <- data.frame(
  week.day = rep(" ", length(testData)),
  hour = rep(" ", length(testData)),
  
  depDelayLeeds = rep(0, length(testData)),
  depDelayWake = rep(0, length(testData)),
  depDelayBarn = rep(0, length(testData)),
  depDelayMead = rep(0, length(testData)),
  
  arrDelayWake = rep(0, length(testData)),
  arrDelayBarn = rep(0, length(testData)),
  arrDelayMead = rep(0, length(testData)),
  arrDelayShef = rep(0, length(testData)),
  arrDelayNotts = rep(0, length(testData)),
  stringsAsFactors = FALSE
)

# Loop through all the values of Training Data to fill the Matrix Data Variable
for (i in 1:length(testData)) {
  dummy <- testData[[i]]
  testMatrixData$week.day[i] <- dummy$timings$day.week[1]
  testMatrixData$hour[i] <- dummy$congestion$hour[1]
  
  depLeeds <- textToSeconds(tail(dummy$timings$departure.time,4)[1])
  schLeeds <- textToSeconds(tail(dummy$timings$departure.schedule,4)[1])
  testMatrixData$depDelayLeeds[i] <- depLeeds - schLeeds
  
  depWake <- textToSeconds(tail(dummy$timings$departure.time,3)[1])
  schWake <- textToSeconds(tail(dummy$timings$departure.schedule,3)[1])
  testMatrixData$depDelayWake[i] <- depWake - schWake
  
  depBarn <- textToSeconds(tail(dummy$timings$departure.time,2)[1])
  schBarn <- textToSeconds(tail(dummy$timings$departure.schedule,2)[1])
  testMatrixData$depDelayBarn[i] <- depBarn - schBarn
  
  depMead <- textToSeconds(tail(dummy$timings$departure.time,1))
  schMead <- textToSeconds(tail(dummy$timings$departure.schedule,1))
  testMatrixData$depDelayMead[i] <- depMead - schMead
  
  arrWake <- textToSeconds(tail(dummy$timings$arrival.time,4)[1])
  schWake <- textToSeconds(tail(dummy$timings$arrival.schedule,4)[1])
  testMatrixData$arrDelayWake[i] <- arrWake - schWake
  
  arrBarn <- textToSeconds(tail(dummy$timings$arrival.time,3)[1])
  schBarn <- textToSeconds(tail(dummy$timings$arrival.schedule,3)[1])
  testMatrixData$arrDelayBarn[i] <- arrBarn - schBarn
  
  arrMead <- textToSeconds(tail(dummy$timings$arrival.time,2)[1])
  schMead <- textToSeconds(tail(dummy$timings$arrival.schedule,2)[1])
  testMatrixData$arrDelayMead[i] <- arrMead - schMead
  
  arrSheffield <- textToSeconds(tail(dummy$timings$arrival.time,1))
  schSheffield <- textToSeconds(tail(dummy$timings$arrival.schedule,1))
  testMatrixData$arrDelayShef[i] <- arrSheffield - schSheffield
  
  testMatrixData$arrDelayNotts[i] <- dummy$arrival$delay.secs[1]
}

print(testData[[1]])

head(testMatrixData)

dim(testMatrixData)


# Loop through each instance in Training Data
for (i in seq_along(testData)) {
  testData[[i]]$congestion <- testData[[i]]$congestion %>%
    left_join(historicalCongestion, by = c("week.day" = "Day", "hour" = "Hour")) %>%
    mutate(
      Leeds_train_diff = (testData[[i]]$congestion$Leeds.trains - Leeds.trains.y) / Leeds.trains.y,
      Sheff_train_diff = (testData[[i]]$congestion$Sheffield.trains - Sheffield.trains.y) / Sheffield.trains.y,
      Notts_train_diff = (testData[[i]]$congestion$Nottingham.trains - Nottingham.trains.y) / Nottingham.trains.y,
      
      Leeds_delay_diff = (testData[[i]]$congestion$Leeds.av.delay - Leeds.av.delay.y), # / Leeds.av.delay.y,
      Sheff_delay_diff = (testData[[i]]$congestion$Sheffield.av.delay - Sheffield.av.delay.y), # / Sheffield.av.delay.y,
      Notts_delay_diff = (testData[[i]]$congestion$Nottingham.av.delay - Nottingham.av.delay.y), # / Nottingham.av.delay.y,
      
      high_congestion = ifelse(testData[[i]]$congestion$Leeds.trains > Leeds.trains.y, 1, 0),
      high_delay = ifelse(testData[[i]]$congestion$Leeds.av.delay > Leeds.av.delay.y + 2, 1, 0)  # 2 sec Threshold
    ) %>%
    select(-ends_with(".y"))  # Remove duplicate columns after join
}


# Initialize empty dataframe to store extracted features
congestion_features <- data.frame()

# Loop through testData to extract features
for (i in seq_along(testData)) {
  if (!is.null(testData[[i]]$congestion)) {
    temp_data <- testData[[i]]$congestion %>%
      select(week.day, hour, Leeds_train_diff, Sheff_train_diff, Notts_train_diff, 
             Leeds_delay_diff, Sheff_delay_diff, Notts_delay_diff, high_congestion, high_delay)
    
    # Append extracted features to congestion_features
    congestion_features <- bind_rows(congestion_features, temp_data)
  }
}


# Ensure `hour` is an integer in both datasets
testMatrixData <- testMatrixData %>%
  mutate(hour = as.integer(hour))

congestion_features <- congestion_features %>%
  mutate(hour = as.integer(hour))

testMatrixData$hour_sin <- sin(2 * pi * testMatrixData$hour / 24)
testMatrixData$hour_cos <- cos(2 * pi * testMatrixData$hour / 24)


if (nrow(testMatrixData) == nrow(congestion_features)) {
  # Append all congestion features directly to testMatrixData
  testMatrixData <- cbind(testMatrixData, congestion_features)
  
  cat("Congestion features successfully added to testMatrixData! \n")
} else {
  cat("Error: Row count mismatch between matrixData and congestion_features \n")
}

# Same weekday/weekend binary encoding as the training set
testMatrixData$day_encoded <- ifelse(testMatrixData$week.day %in% c("Saturday", "Sunday"), 1, 0)

head(testMatrixData)

dim(testMatrixData)


testMatrixData <- testMatrixData %>%
  select(-c(week.day, hour))

head(testMatrixData)

dim(testMatrixData)


# Step 2 -> Feature Selection

testMatrixData <- testMatrixData[, !colnames(testMatrixData) %in% c("high_congestion", "high_delay", "day_encoded")]
colnames(testMatrixData)

# Check missing values in each column
colSums(is.na(testMatrixData))

dim(testMatrixData)

# Remove Target Variable
testMatrixData <- testMatrixData[, colnames(testMatrixData) != "arrDelayNotts"]

# 1. Departure Delay Leeds -> [17]
testMatrixData$abs_col_1 <- abs(testMatrixData$depDelayLeeds)

testMatrixData$col_1_sq <- testMatrixData$depDelayLeeds^2
testMatrixData$col_1_cu <- testMatrixData$depDelayLeeds^3

testMatrixData$col_1_Sub_1 <- testMatrixData$depDelayLeeds - testMatrixData$depDelayWake
testMatrixData$col_1_Sub_2 <- testMatrixData$depDelayLeeds - testMatrixData$depDelayBarn
testMatrixData$col_1_Sub_3 <- testMatrixData$depDelayLeeds - testMatrixData$depDelayMead
testMatrixData$col_1_Sub_4 <- testMatrixData$depDelayLeeds - testMatrixData$arrDelayWake
testMatrixData$col_1_Sub_5 <- testMatrixData$depDelayLeeds - testMatrixData$arrDelayBarn
testMatrixData$col_1_Sub_6 <- testMatrixData$depDelayLeeds - testMatrixData$arrDelayMead
testMatrixData$col_1_Sub_7 <- testMatrixData$depDelayLeeds - testMatrixData$arrDelayShef

testMatrixData$col_1_abs_1 <- abs(testMatrixData$col_1_Sub_1)
testMatrixData$col_1_abs_2 <- abs(testMatrixData$col_1_Sub_2)
testMatrixData$col_1_abs_3 <- abs(testMatrixData$col_1_Sub_3)
testMatrixData$col_1_abs_4 <- abs(testMatrixData$col_1_Sub_4)
testMatrixData$col_1_abs_5 <- abs(testMatrixData$col_1_Sub_5)
testMatrixData$col_1_abs_6 <- abs(testMatrixData$col_1_Sub_6)
testMatrixData$col_1_abs_7 <- abs(testMatrixData$col_1_Sub_7)


# 2. Departure Delay Wake -> [15]
testMatrixData$abs_col_2 <- abs(testMatrixData$depDelayWake)

testMatrixData$col_2_sq <- testMatrixData$depDelayWake^2
testMatrixData$col_2_cu <- testMatrixData$depDelayWake^3

testMatrixData$col_2_Sub_1 <- testMatrixData$depDelayWake - testMatrixData$depDelayBarn
testMatrixData$col_2_Sub_2 <- testMatrixData$depDelayWake - testMatrixData$depDelayMead
testMatrixData$col_2_Sub_3 <- testMatrixData$depDelayWake - testMatrixData$arrDelayWake
testMatrixData$col_2_Sub_4 <- testMatrixData$depDelayWake - testMatrixData$arrDelayBarn
testMatrixData$col_2_Sub_5 <- testMatrixData$depDelayWake - testMatrixData$arrDelayMead
testMatrixData$col_2_Sub_6 <- testMatrixData$depDelayWake - testMatrixData$arrDelayShef

testMatrixData$col_2_abs_1 <- abs(testMatrixData$col_2_Sub_1)
testMatrixData$col_2_abs_2 <- abs(testMatrixData$col_2_Sub_2)
testMatrixData$col_2_abs_3 <- abs(testMatrixData$col_2_Sub_3)
testMatrixData$col_2_abs_4 <- abs(testMatrixData$col_2_Sub_4)
testMatrixData$col_2_abs_5 <- abs(testMatrixData$col_2_Sub_5)
testMatrixData$col_2_abs_6 <- abs(testMatrixData$col_2_Sub_6)


# 3. Departure Delay Barn -> [13]
testMatrixData$abs_col_3 <- abs(testMatrixData$depDelayBarn)

testMatrixData$col_3_sq <- testMatrixData$depDelayBarn^2
testMatrixData$col_3_cu <- testMatrixData$depDelayBarn^3

testMatrixData$col_3_Sub_1 <- testMatrixData$depDelayBarn - testMatrixData$depDelayMead
testMatrixData$col_3_Sub_2 <- testMatrixData$depDelayBarn - testMatrixData$arrDelayWake
testMatrixData$col_3_Sub_3 <- testMatrixData$depDelayBarn - testMatrixData$arrDelayBarn
testMatrixData$col_3_Sub_4 <- testMatrixData$depDelayBarn - testMatrixData$arrDelayMead
testMatrixData$col_3_Sub_5 <- testMatrixData$depDelayBarn - testMatrixData$arrDelayShef

testMatrixData$col_3_abs_1 <- abs(testMatrixData$col_3_Sub_1)
testMatrixData$col_3_abs_2 <- abs(testMatrixData$col_3_Sub_2)
testMatrixData$col_3_abs_3 <- abs(testMatrixData$col_3_Sub_3)
testMatrixData$col_3_abs_4 <- abs(testMatrixData$col_3_Sub_4)
testMatrixData$col_3_abs_5 <- abs(testMatrixData$col_3_Sub_5)


# 4. Departure Delay Mead -> [11]
testMatrixData$abs_col_4 <- abs(testMatrixData$depDelayMead)

testMatrixData$col_4_sq <- testMatrixData$depDelayMead^2
testMatrixData$col_4_cu <- testMatrixData$depDelayMead^3

testMatrixData$col_4_Sub_1 <- testMatrixData$depDelayMead - testMatrixData$arrDelayWake
testMatrixData$col_4_Sub_2 <- testMatrixData$depDelayMead - testMatrixData$arrDelayBarn
testMatrixData$col_4_Sub_3 <- testMatrixData$depDelayMead - testMatrixData$arrDelayMead
testMatrixData$col_4_Sub_4 <- testMatrixData$depDelayMead - testMatrixData$arrDelayShef

testMatrixData$col_4_abs_1 <- abs(testMatrixData$col_4_Sub_1)
testMatrixData$col_4_abs_2 <- abs(testMatrixData$col_4_Sub_2)
testMatrixData$col_4_abs_3 <- abs(testMatrixData$col_4_Sub_3)
testMatrixData$col_4_abs_4 <- abs(testMatrixData$col_4_Sub_4)


# 5. Arrival Delay Wake -> [9]
testMatrixData$abs_col_5 <- abs(testMatrixData$arrDelayWake)

testMatrixData$col_5_sq <- testMatrixData$arrDelayWake^2
testMatrixData$col_5_cu <- testMatrixData$arrDelayWake^3

testMatrixData$col_5_Sub_1 <- testMatrixData$arrDelayWake - testMatrixData$arrDelayBarn
testMatrixData$col_5_Sub_2 <- testMatrixData$arrDelayWake - testMatrixData$arrDelayMead
testMatrixData$col_5_Sub_3 <- testMatrixData$arrDelayWake - testMatrixData$arrDelayShef

testMatrixData$col_5_abs_1 <- abs(testMatrixData$col_5_Sub_1)
testMatrixData$col_5_abs_2 <- abs(testMatrixData$col_5_Sub_2)
testMatrixData$col_5_abs_3 <- abs(testMatrixData$col_5_Sub_3)


# 6. Arrival Delay Barn -> [7]
testMatrixData$abs_col_6 <- abs(testMatrixData$arrDelayBarn)

testMatrixData$col_6_sq <- testMatrixData$arrDelayBarn^2
testMatrixData$col_6_cu <- testMatrixData$arrDelayBarn^3

testMatrixData$col_6_Sub_1 <- testMatrixData$arrDelayBarn - testMatrixData$arrDelayMead
testMatrixData$col_6_Sub_2 <- testMatrixData$arrDelayBarn - testMatrixData$arrDelayShef

testMatrixData$col_6_abs_1 <- abs(testMatrixData$col_5_Sub_1)
testMatrixData$col_6_abs_2 <- abs(testMatrixData$col_5_Sub_2)


# 7. Arrival Delay Mead -> [5]
testMatrixData$abs_col_7 <- abs(testMatrixData$arrDelayMead)

testMatrixData$col_7_sq <- testMatrixData$arrDelayMead^2
testMatrixData$col_7_cu <- testMatrixData$arrDelayMead^3

testMatrixData$col_7_Sub_1 <- testMatrixData$arrDelayMead - testMatrixData$arrDelayShef

testMatrixData$col_7_abs_1 <- abs(testMatrixData$col_5_Sub_1)


# 8. Arrival Delay Shef -> [3]
testMatrixData$abs_col_8 <- abs(testMatrixData$arrDelayShef)

testMatrixData$col_8_sq <- testMatrixData$arrDelayShef^2
testMatrixData$col_8_cu <- testMatrixData$arrDelayShef^3


# Hour in Sin & Cos -> [7]
testMatrixData$hour_sin_cos <- testMatrixData$hour_sin * testMatrixData$hour_cos

testMatrixData$hour_sin_sq <- testMatrixData$hour_sin^2
testMatrixData$hour_sin_cu <- testMatrixData$hour_sin^3

testMatrixData$hour_cos_sq <- testMatrixData$hour_cos^2
testMatrixData$hour_cos_cu <- testMatrixData$hour_cos^3

testMatrixData$hour_sin_cos_sq <- testMatrixData$hour_sin_cos^2
testMatrixData$hour_sin_cos_cu <- testMatrixData$hour_sin_cos^3


# 9. Train Difference -> [12]
testMatrixData$train_leeds_sq <- testMatrixData$Leeds_train_diff^2
testMatrixData$train_leeds_cu <- testMatrixData$Leeds_train_diff^3

testMatrixData$train_sheff_sq <- testMatrixData$Sheff_train_diff^2
testMatrixData$train_sheff_cu <- testMatrixData$Sheff_train_diff^3

testMatrixData$train_notts_sq <- testMatrixData$Notts_train_diff^2
testMatrixData$train_notts_cu <- testMatrixData$Notts_train_diff^3

testMatrixData$col_9_Sub_1 <- testMatrixData$Leeds_train_diff - testMatrixData$Sheff_train_diff
testMatrixData$col_9_Sub_2 <- testMatrixData$Leeds_train_diff - testMatrixData$Notts_train_diff
testMatrixData$col_9_Sub_3 <- testMatrixData$Sheff_train_diff - testMatrixData$Notts_train_diff

testMatrixData$col_9_abs_1 <- abs(testMatrixData$col_9_Sub_1)
testMatrixData$col_9_abs_2 <- abs(testMatrixData$col_9_Sub_2)
testMatrixData$col_9_abs_3 <- abs(testMatrixData$col_9_Sub_3)


# 10. Delay Difference -> [12]
testMatrixData$delay_leeds_sq <- testMatrixData$Leeds_delay_diff^2
testMatrixData$delay_leeds_cu <- testMatrixData$Leeds_delay_diff^3

testMatrixData$delay_sheff_sq <- testMatrixData$Sheff_delay_diff^2
testMatrixData$delay_sheff_cu <- testMatrixData$Sheff_delay_diff^3

testMatrixData$delay_notts_sq <- testMatrixData$Notts_delay_diff^2
testMatrixData$delay_notts_cu <- testMatrixData$Notts_delay_diff^3

testMatrixData$col_10_Sub_1 <- testMatrixData$Leeds_delay_diff - testMatrixData$Sheff_delay_diff
testMatrixData$col_10_Sub_2 <- testMatrixData$Leeds_delay_diff - testMatrixData$Notts_delay_diff
testMatrixData$col_10_Sub_3 <- testMatrixData$Sheff_delay_diff - testMatrixData$Notts_delay_diff

testMatrixData$col_10_abs_1 <- abs(testMatrixData$col_10_Sub_1)
testMatrixData$col_10_abs_2 <- abs(testMatrixData$col_10_Sub_2)
testMatrixData$col_10_abs_3 <- abs(testMatrixData$col_10_Sub_3)


# Train & Delay Interaction -> [6] Total = 117
testMatrixData$train_leeds_interaction <- testMatrixData$Leeds_train_diff * testMatrixData$depDelayLeeds
testMatrixData$train_sheff_interaction <- testMatrixData$Sheff_train_diff * testMatrixData$arrDelayShef
testMatrixData$train_notts_interaction <- testMatrixData$Notts_train_diff * testMatrixData$arrDelayShef

testMatrixData$delay_leeds_interaction <- testMatrixData$Leeds_delay_diff * testMatrixData$depDelayLeeds
testMatrixData$delay_sheff_interaction <- testMatrixData$Sheff_delay_diff * testMatrixData$arrDelayShef
testMatrixData$delay_notts_interaction <- testMatrixData$Notts_delay_diff * testMatrixData$arrDelayShef

testMatrixData <- testMatrixData[, selected_features]

colnames(testMatrixData)

dim(X_train)

dim(testMatrixData)


# Step 3 -> Imputing Missing Values

# Check missing values in each column
colSums(is.na(testMatrixData))

# NOTE (known gap): the loop that applies the training-set column means (mean_values,
# computed earlier) to any NAs in testMatrixData is disabled below. If colSums() above
# shows non-zero counts for any column, uncomment this before predicting on that column,
# otherwise xgboost/predict.gam will error or silently drop those rows.
# for (col in colnames(testMatrixData)) {
#   if (col %in% names(mean_values)) {
#     testMatrixData[[col]][is.na(testMatrixData[[col]])] <- mean_values[[col]]
#   }
# }

# Step 4 -> PCA was not used in the final model (see note in the training section above),
# so no PCA transform is applied to the test set either.

# Step 5 -> Prediction

# Convert testMatrixData to matrix format
X_test_final <- data.matrix(testMatrixData) 

# Make predictions using trained XGBoost model
xgb_predictions  <- predict(xgb_model, X_test_final)

dim(xgb_predictions)

xgb_predictions <- data.frame(Predictions = xgb_predictions)

dim(xgb_predictions)


# Make predictions using trained GAM model
gam_predictions <- predict(best_gam_model, newdata = testMatrixData)

dim(gam_predictions)

gam_predictions <- data.frame(Predictions = gam_predictions)

dim(gam_predictions)


# Model Blending (Ensemble) - Combine XGBoost & GAM Predictions
final_predictions <- (0.2 * xgb_predictions) + (0.8 * gam_predictions)

# Print some sample predictions
head(final_predictions)

dim(final_predictions)

final_predictions <- data.frame(Predictions = final_predictions)

dim(final_predictions)


# Uncomment to save predictions to disk:
# write.csv(final_predictions, file = "Blended.csv", row.names = FALSE)

# ----------------------------------------------------------------------------------------
