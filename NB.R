# --- Replicable Naive Bayes Modeling Pipeline for PM10 Prediction ---

# Required libraries
# Uncomment the install.packages lines if the packages are not already installed.
# install.packages(c("rgdal", "raster", "plyr", "dplyr", "RStoolbox", "RColorBrewer", "ggplot2",
#                    "sp", "caret", "doParallel", "e1071", "pdftools", "gridExtra", "GGally",
#                    "doSNOW", "klaR", "pROC"))

library(rgdal)         # spatial data processing
library(raster)        # raster processing
library(plyr)          # data manipulation
library(dplyr)         # data manipulation
library(RStoolbox)     # image analysis & plotting spatial data
library(RColorBrewer)  # color palettes
library(ggplot2)       # plotting
library(sp)            # spatial data
library(caret)         # machine learning
library(doParallel)    # parallel processing
library(e1071)         # Naive Bayes
library(pdftools)      # PDF tools
library(gridExtra)     # arranging multiple plots
library(GGally)        # advanced plotting
library(doSNOW)        # parallel computing for caret
library(klaR)          # additional NB support
library(pROC)          # ROC curves

# Define relative paths for data and outputs
data_dir    <- "data"  # folder containing S2.csv, Levels_key.csv, etc.
rasters_dir <- file.path(data_dir, "rasters")  # folder with all raster layers
output_dir  <- "output"  # folder for saving plots and maps

# Ensure output directory exists
if (!dir.exists(output_dir)) dir.create(output_dir)

######################
#### 1. Data Preparation ####
######################

# Read the dataset (CSV file should be placed in the 'data' folder)
data_file <- file.path(data_dir, "S2.csv")
original <- read.csv(data_file, header = TRUE, stringsAsFactors = FALSE)
original <- na.omit(original)
original <- data.frame(original)

# Select only the required variables (as per publication)
selected_vars <- c("LevelAve", "Atmospher_density", "Elevation", "LST", "Nightlit",
                   "Rainfall", "Road_density", "SAVI", "Slope", "TRI", "TWI",
                   "Tempreature", "UI", "WEX", "buld_density", "dist_Airport",
                   "dist_Construction", "dist_Petrolstation", "dist_Railway",
                   "dist_Road", "dist_Unionmarket", "dist_Wasteplate",
                   "dist_busstation", "dist_industry", "dist_qurry", "wind_speed")
original <- original[selected_vars]
original$LevelAve <- as.factor(original$LevelAve)

# Define normalization function
normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# Normalize predictors (columns 2 to 26) and append the response variable
original_norm <- as.data.frame(lapply(original[, 2:26], normalize))
original_norm$PM <- original$LevelAve

# Split dataset into training (70%) and testing (30%) sets with a fixed seed for replication
set.seed(123)
data_index <- sample(1:nrow(original), size = 0.70 * nrow(original), replace = FALSE)
train_data <- original_norm[data_index, ]
test_data  <- original_norm[-data_index, ]
train_labels <- original[data_index, "LevelAve"]
test_labels  <- original[-data_index, "LevelAve"]

######################
#### 2. Data Visualization ####
######################

# Visualize relationships between predictors and response; save plots to a PDF
pdf(file = file.path(output_dir, "Variable_Relationships.pdf"))
variables <- selected_vars[-1]  # exclude the response variable

plots <- list()
plots_per_page <- 4
num_pages <- ceiling(length(variables) / plots_per_page)

for (i in 1:num_pages) {
  start_idx <- (i - 1) * plots_per_page + 1
  end_idx   <- min(i * plots_per_page, length(variables))
  page_plots <- list()
  for (j in start_idx:end_idx) {
    var_name <- variables[j]
    p <- ggplot(original, aes_string(x = var_name, colour = "LevelAve")) +
      geom_freqpoly(binwidth = 1) +
      labs(title = paste(var_name, "Distribution by PM10 levels"))
    page_plots[[var_name]] <- p
  }
  do.call(grid.arrange, c(page_plots, ncol = 2))
}
dev.off()

# Analyze "perfect splits" across features—a crucial replication step
number_perfect_splits <- apply(original_norm, 2, function(col) {
  t <- table(original$LevelAve, col)
  sum(t == 0)
})
order_idx <- order(number_perfect_splits, decreasing = TRUE)
number_perfect_splits <- number_perfect_splits[order_idx]

# Plot and save the perfect splits barplot
pdf(file = file.path(output_dir, "Perfect_Splits_Comparison.pdf"))
par(mar = c(10, 4, 2, 2))
bar_colors <- colorRampPalette(c("lightblue", "steelblue"))(length(number_perfect_splits))
barplot(number_perfect_splits,
        main = "Perfect Splits Across Features",
        xlab = "Number of Perfect Splits",
        ylab = "Features",
        las = 2,
        col = bar_colors,
        beside = TRUE)
dev.off()

######################
#### 3. Modeling ####
######################

# Set up parallel processing for model training
cl <- makeCluster(detectCores())
registerDoParallel(cl)

# Define training control using repeated 10-fold cross-validation (3 repeats)
train_ctrl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 3,
                           returnResamp = "all",
                           allowParallel = TRUE)

# Train the default Naive Bayes model
set.seed(849)
fit_nb_def <- train(PM ~ Atmospher_density + Elevation + LST + Nightlit + Rainfall +
                      Road_density + SAVI + Slope + TRI + TWI + Tempreature +
                      UI + WEX + buld_density + dist_Airport + dist_Construction +
                      dist_Petrolstation + dist_Railway + dist_Road + dist_Unionmarket +
                      dist_Wasteplate + dist_busstation + dist_industry + dist_qurry +
                      wind_speed,
                    data = train_data,
                    method = "nb",
                    metric = "Accuracy",
                    preProc = c("center", "scale"),
                    trControl = train_ctrl)

# Output default model performance and variable importance
print(fit_nb_def$resample)
var_imp_def <- varImp(fit_nb_def)
plot(var_imp_def)

pred_def <- predict(fit_nb_def, test_data, type = "raw")
cm_def <- confusionMatrix(pred_def, test_labels)
print(cm_def)

# Parameter tuning: explore different Laplace corrections and bandwidth adjustments
tune_grid <- expand.grid(fL = c(0, 0.5, 1.0),
                         usekernel = TRUE,
                         adjust = c(0, 0.5, 1.0))
set.seed(849)
fit_nb_tuned <- train(PM ~ Atmospher_density + Elevation + LST + Nightlit + Rainfall +
                        Road_density + SAVI + Slope + TRI + TWI + Tempreature +
                        UI + WEX + buld_density + dist_Airport + dist_Construction +
                        dist_Petrolstation + dist_Railway + dist_Road + dist_Unionmarket +
                        dist_Wasteplate + dist_busstation + dist_industry + dist_qurry +
                        wind_speed,
                      data = train_data,
                      method = "nb",
                      tuneGrid = tune_grid,
                      metric = "Accuracy",
                      preProc = c("center", "scale"),
                      trControl = train_ctrl,
                      importance = TRUE)
print(fit_nb_tuned$results)
pred_tuned <- predict(fit_nb_tuned, test_data, type = "raw")
cm_tuned <- confusionMatrix(pred_tuned, test_labels)
print(cm_tuned)

# Further tuning using best hyperparameters (e.g., fL = 0, adjust = 0.5)
tune_grid_best <- expand.grid(fL = 0,
                              usekernel = TRUE,
                              adjust = 0.5)
set.seed(849)
fit_nb_best <- train(PM ~ Atmospher_density + Elevation + LST + Nightlit + Rainfall +
                       Road_density + SAVI + Slope + TRI + TWI + Tempreature +
                       UI + WEX + buld_density + dist_Airport + dist_Construction +
                       dist_Petrolstation + dist_Railway + dist_Road + dist_Unionmarket +
                       dist_Wasteplate + dist_busstation + dist_industry + dist_qurry +
                       wind_speed,
                     data = train_data,
                     method = "nb",
                     tuneGrid = tune_grid_best,
                     metric = "Accuracy",
                     preProc = c("center", "scale"),
                     trControl = train_ctrl,
                     importance = TRUE)
print(fit_nb_best$results)
pred_best <- predict(fit_nb_best, test_data, type = "raw")
cm_best <- confusionMatrix(pred_best, test_labels)
print(cm_best)

# Generate ROC curves using probability predictions from the best model
prob_predictions <- predict(fit_nb_best, test_data, type = "prob")
prob_predictions$observed <- test_labels

roc_Moderate  <- roc(ifelse(prob_predictions$observed == "Moderate", "Moderate", "non-Moderate"),
                     as.numeric(prob_predictions$Moderate))
roc_Good      <- roc(ifelse(prob_predictions$observed == "Good", "Good", "non-Good"),
                     as.numeric(prob_predictions$Good))
roc_UHealSesn <- roc(ifelse(prob_predictions$observed == "UHealSesn", "UHealSesn", "non-UHealSesn"),
                     as.numeric(prob_predictions$UHealSesn))

pdf(file = file.path(output_dir, "ROC_Curves.pdf"))
plot(roc_Moderate, col = "blue", main = "ROC Curves: NB Best Model", xlim = c(0.61, 0.1))
lines(roc_UHealSesn, col = "red")
lines(roc_Good, col = "green")
legend("topleft",
       legend = c(paste("Good AUC =", round(roc_Good$auc, 2)),
                  paste("Moderate AUC =", round(roc_Moderate$auc, 2)),
                  paste("UHealSesn AUC =", round(roc_UHealSesn$auc, 2))),
       fill = c("green", "blue", "red"),
       inset = 0.42)
dev.off()

# Train the final model on all available data
original_norm$PM <- as.factor(original$LevelAve)
set.seed(849)
fit_nb_all <- train(PM ~ Atmospher_density + Elevation + LST + Nightlit + Rainfall +
                      Road_density + SAVI + Slope + TRI + TWI + Tempreature +
                      UI + WEX + buld_density + dist_Airport + dist_Construction +
                      dist_Petrolstation + dist_Railway + dist_Road + dist_Unionmarket +
                      dist_Wasteplate + dist_busstation + dist_industry + dist_qurry +
                      wind_speed,
                    data = original_norm,
                    method = "nb",
                    metric = "Accuracy",
                    tuneGrid = tune_grid_best,
                    preProc = c("center", "scale"),
                    trControl = train_ctrl)
var_imp_all <- varImp(fit_nb_all)
plot(var_imp_all)

# Stop parallel processing
stopCluster(cl)

######################
#### 4. Produce Prediction Maps Using Raster Data ####
######################

# Load raster layers (files must reside in the 'data/rasters' folder)
elevation <- raster(file.path(rasters_dir, "Elevation.tif"))
names(elevation) <- "Elevation"

rainfall <- raster(file.path(rasters_dir, "Rainfall.tif"))
names(rainfall) <- "Rainfall"
rainfall <- resample(rainfall, elevation, method = "bilinear")

nightlit <- raster(file.path(rasters_dir, "Nightlit.tif"))
names(nightlit) <- "Nightlit"
nightlit <- resample(nightlit, elevation, method = "bilinear")

tempreature <- raster(file.path(rasters_dir, "Tempreature.tif"))
names(tempreature) <- "Tempreature"
tempreature <- resample(tempreature, elevation, method = "bilinear")

# Load additional raster layers as needed...
# For example: Atmospher_density, LST, Road_density, SAVI, Slope, TRI, TWI,
# UI, WEX, buld_density, dist_Airport, dist_Construction, dist_Petrolstation,
# dist_Railway, dist_Road, dist_Unionmarket, dist_Wasteplate, dist_busstation,
# dist_industry, dist_qurry, wind_speed

# Stack all predictor rasters
rasters_stack <- stack(
  Atmospher_density = raster(file.path(rasters_dir, "Atmospher_density.tif")),
  Elevation         = elevation,
  LST               = raster(file.path(rasters_dir, "LST.tif")),
  Nightlit          = nightlit,
  Rainfall          = rainfall,
  Road_density      = raster(file.path(rasters_dir, "Road_density.tif")),
  SAVI              = raster(file.path(rasters_dir, "SAVI.tif")),
  Slope             = raster(file.path(rasters_dir, "Slope.tif")),
  TRI               = raster(file.path(rasters_dir, "TRI.tif")),
  TWI               = raster(file.path(rasters_dir, "TWI.tif")),
  Tempreature       = tempreature,
  UI                = raster(file.path(rasters_dir, "UI.tif")),
  WEX               = raster(file.path(rasters_dir, "WEX.tif")),
  buld_density      = raster(file.path(rasters_dir, "buld_density.tif")),
  dist_Airport      = raster(file.path(rasters_dir, "dist_Airport.tif")),
  dist_Construction = raster(file.path(rasters_dir, "dist_Construction.tif")),
  dist_Petrolstation= raster(file.path(rasters_dir, "dist_Petrolstation.tif")),
  dist_Railway      = raster(file.path(rasters_dir, "dist_Railway.tif")),
  dist_Road         = raster(file.path(rasters_dir, "dist_Road.tif")),
  dist_Unionmarket  = raster(file.path(rasters_dir, "dist_Unionmarket.tif")),
  dist_Wasteplate   = raster(file.path(rasters_dir, "dist_Wasteplate.tif")),
  dist_busstation   = raster(file.path(rasters_dir, "dist_busstation.tif")),
  dist_industry     = raster(file.path(rasters_dir, "dist_industry.tif")),
  dist_qurry        = raster(file.path(rasters_dir, "dist_qurry.tif")),
  wind_speed        = raster(file.path(rasters_dir, "wind_speed.tif"))
)

# Convert raster stack to dataframe for prediction
rasters_df <- as.data.frame(rasters_stack, xy = TRUE, na.rm = TRUE)
predictors_raster <- rasters_df[, -c(1,2)]
predictors_raster_norm <- as.data.frame(lapply(predictors_raster, normalize))

# Produce probability predictions using the final model
prob_pred <- predict(fit_nb_all, predictors_raster_norm, type = "prob")
rasters_df$Levels_Moderate  <- prob_pred$Moderate
rasters_df$Levels_Good      <- prob_pred$Good
rasters_df$Levels_UHealSesn <- prob_pred$UHealSesn

# Convert probability predictions to raster maps
r_good      <- rasterFromXYZ(rasters_df[, c("x", "y", "Levels_Good")])
crs(r_good) <- projection(elevation)
r_moderate  <- rasterFromXYZ(rasters_df[, c("x", "y", "Levels_Moderate")])
crs(r_moderate) <- projection(elevation)
r_uhealsesn <- rasterFromXYZ(rasters_df[, c("x", "y", "Levels_UHealSesn")])
crs(r_uhealsesn) <- projection(elevation)

# Save probability maps to the output folder
writeRaster(r_moderate,  file.path(output_dir, "Prediction_NB_Moderate.tif"),
            format = "GTiff", overwrite = TRUE)
writeRaster(r_uhealsesn, file.path(output_dir, "Prediction_NB_UHealSesn.tif"),
            format = "GTiff", overwrite = TRUE)
writeRaster(r_good,      file.path(output_dir, "Prediction_NB_Good.tif"),
            format = "GTiff", overwrite = TRUE)

# Produce classification predictions
class_pred <- predict(fit_nb_all, predictors_raster_norm, type = "raw")
rasters_df$LevelAve <- class_pred

# Read the levels key (ensure Levels_key.csv is in the data folder)
levels_key_file <- file.path(data_dir, "Levels_key.csv")
levels_key <- read.csv(levels_key_file, header = TRUE)
# Merge the predictions with levels key using plyr join
grid_joined <- join(rasters_df, levels_key, by = "LevelAve", type = "inner")
grid_joined <- na.omit(grid_joined)
classification_raster <- rasterFromXYZ(grid_joined[, c("x", "y", "Level_ID")])
crs(classification_raster) <- projection(elevation)
writeRaster(classification_raster, file.path(output_dir, "Classification_Map_NB.tif"),
            format = "GTiff", overwrite = TRUE)

# End of pipeline