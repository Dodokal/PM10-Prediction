##### KNN Model for PM10 Prediction

# Load required packages
library(caret)         # Machine learning
library(rgdal)         # Spatial data processing
library(raster)        # Raster processing
library(plyr)          # Data manipulation
library(dplyr)         # Data manipulation
library(RStoolbox)     # Plotting spatial data
library(RColorBrewer)  # Color palettes
library(ggplot2)       # Plotting
library(sp)            # Spatial data
library(doParallel)    # Parallel processing
library(doSNOW)        # Parallel processing via SNOW
library(e1071)         # SVM utilities (required by caret)
library(pROC)          # ROC analysis

# -----------------------------
# Step 1: Import and Clean Data
# -----------------------------
# Read the CSV file (assumed to be in the repository root)
original <- read.csv("./S2.csv", header = TRUE, stringsAsFactors = FALSE)
original <- na.omit(original)
original <- data.frame(original)  # Remove unwanted attributes

# Select relevant variables and convert response to a factor
original <- original[c("LevelAve", "Atmospher_density", "Elevation", "LST", "Nightlit",
                       "Rainfall", "Road_density", "SAVI", "Slope", "TRI", "TWI",
                       "Tempreature", "UI", "WEX", "buld_density", "dist_Airport",
                       "dist_Construction", "dist_Petrolstation", "dist_Railway",
                       "dist_Road", "dist_Unionmarket", "dist_Wasteplate", "dist_busstation",
                       "dist_industry", "dist_qurry", "wind_speed")]
original$LevelAve <- as.factor(original$LevelAve)
print(summary(original))

# -----------------------------
# Step 2: Data Normalization
# -----------------------------
normalize <- function(x) { (x - min(x)) / (max(x) - min(x)) }
# Normalize predictor variables (columns 2 to 26)
original.n <- as.data.frame(lapply(original[, 2:26], normalize))
# Append the response variable to the normalized dataframe
original.n$PM <- original$LevelAve

# -----------------------------
# Step 3: Data Splitting
# -----------------------------
set.seed(123)
# Use 70% of the data for training and the remaining 30% for testing
data.d <- sample(1:nrow(original), size = nrow(original) * 0.70, replace = FALSE)
train.data <- original.n[data.d, ]
test.data  <- original.n[-data.d, ]

# Create target labels for training and testing sets
train.data_labels <- original[data.d, 1]
test.data_labels  <- original[-data.d, 1]
train.data$PM <- train.data_labels

# -----------------------------
# Step 4: KNN Model Training and Evaluation
# -----------------------------
# Define cross-validation control
control <- trainControl(method = 'repeatedcv', number = 10, repeats = 3)

# (a) Grid Search for Optimal k
set.seed(1)
knn_grid <- train(PM ~ Atmospher_density + Elevation + LST + Nightlit + Rainfall +
                    Road_density + SAVI + Slope + TRI + TWI + Tempreature + UI + WEX +
                    buld_density + dist_Airport + dist_Construction + dist_Petrolstation +
                    dist_Railway + dist_Road + dist_Unionmarket + dist_Wasteplate +
                    dist_busstation + dist_industry + dist_qurry + wind_speed,
                  data = train.data,
                  method = "knn",
                  trControl = control,
                  tuneGrid = expand.grid(k = seq(1, 31, by = 2)))
plot(knn_grid, main = "KNN: Varying k values")
plot(varImp(knn_grid))

# Evaluate the grid-searched model on the test data
p1_knn_grid <- predict(knn_grid, test.data, type = "raw")
print(confusionMatrix(p1_knn_grid, as.factor(test.data_labels)))

# (b) Default KNN Model
set.seed(1)
knn_default <- train(PM ~ Atmospher_density + Elevation + LST + Nightlit + Rainfall +
                       Road_density + SAVI + Slope + TRI + TWI + Tempreature + UI + WEX +
                       buld_density + dist_Airport + dist_Construction + dist_Petrolstation +
                       dist_Railway + dist_Road + dist_Unionmarket + dist_Wasteplate +
                       dist_busstation + dist_industry + dist_qurry + wind_speed,
                     data = train.data,
                     method = "knn",
                     trControl = control)
print(knn_default)
plot(knn_default)
plot(varImp(knn_default), main = "KNN Default Variable Importance")

# Evaluate the default model on the test data
p1_knn_default <- predict(knn_default, test.data, type = "raw")
print(confusionMatrix(p1_knn_default, as.factor(test.data_labels)))
# (Note: The default model provided the best performance.)

# -----------------------------
# Step 5: ROC Curve Analysis
# -----------------------------
# Predict class probabilities for ROC analysis using the default model
predictions <- as.data.frame(predict(knn_default, test.data, type = "prob"))
# Determine the predicted class based on the highest probability
predictions$predict <- colnames(predictions)[apply(predictions[, 1:3], 1, which.max)]
predictions$observed <- test.data_labels
head(predictions)

# Compute ROC curves for each class (e.g., "Good", "Moderate", "UHealSesn")
roc_Good <- roc(ifelse(predictions$observed == "Good", "Good", "non-Good"), as.numeric(predictions$Good))
roc_Moderate <- roc(ifelse(predictions$observed == "Moderate", "Moderate", "non-Moderate"), as.numeric(predictions$Moderate))
roc_UHealSesn <- roc(ifelse(predictions$observed == "UHealSesn", "UHealSesn", "non-UHealSesn"), as.numeric(predictions$UHealSesn))

plot(roc_Moderate, col = "blue", main = "KNN ROC Curve", xlim = c(0.64, 0.1))
lines(roc_UHealSesn, col = "red")
lines(roc_Good, col = "green")
results <- c("UHealSesn AUC" = roc_UHealSesn$auc,
             "Moderate AUC" = roc_Moderate$auc,
             "Good AUC" = roc_Good$auc)
print(results)
legend("topleft", legend = c(paste("UHealSesn AUC =", round(roc_UHealSesn$auc, 2)),
                             paste("Moderate AUC =", round(roc_Moderate$auc, 2)),
                             paste("Good AUC =", round(roc_Good$auc, 2))),
       fill = c("red", "blue", "green"), inset = 0.42)

# -----------------------------
# Step 6: Train Model Using All Data for Spatial Prediction
# -----------------------------
# Append response variable to the complete normalized dataset
original.n$PM <- original$LevelAve
set.seed(849)
fit.KNNAll <- train(PM ~ Atmospher_density + Elevation + LST + Nightlit + Rainfall +
                      Road_density + SAVI + Slope + TRI + TWI + Tempreature + UI + WEX +
                      buld_density + dist_Airport + dist_Construction + dist_Petrolstation +
                      dist_Railway + dist_Road + dist_Unionmarket + dist_Wasteplate +
                      dist_busstation + dist_industry + dist_qurry + wind_speed,
                    data = original.n,
                    method = "knn",
                    trControl = control)
print(varImp(fit.KNNAll))
plot(varImp(fit.KNNAll), main = "KNN Model on All Data")

# -----------------------------
# Step 7: Produce Prediction Maps Using Raster Data
# -----------------------------
# Load raster files (assumed to be in the relative folder "data/resampled")
Atmospher_density <- raster("./data/resampled/Atmospher_density.tif")
names(Atmospher_density) <- "Atmospher_density"

dist_busstation <- raster("./data/resampled/dist_busstation.tif")
names(dist_busstation) <- "dist_busstation"

dist_Railway <- raster("./data/resampled/dist_Railway.tif")
names(dist_Railway) <- "dist_Railway"

dist_Unionmarket <- raster("./data/resampled/dist_Unionmarket.tif")
names(dist_Unionmarket) <- "dist_Unionmarket"

Road_density <- raster("./data/resampled/Road_density.tif")
names(Road_density) <- "Road_density"

dist_Airport <- raster("./data/resampled/dist_Airport.tif")
names(dist_Airport) <- "dist_Airport"

buld_density <- raster("./data/resampled/buld_density.tif")
names(buld_density) <- "buld_density"

dist_Construction <- raster("./data/resampled/dist_Construction.tif")
names(dist_Construction) <- "dist_Construction"

dist_industry <- raster("./data/resampled/dist_industry.tif")
names(dist_industry) <- "dist_industry"

dist_Petrolstation <- raster("./data/resampled/dist_Petrolstation.tif")
names(dist_Petrolstation) <- "dist_Petrolstation"

dist_qurry <- raster("./data/resampled/dist_qurry.tif")
names(dist_qurry) <- "dist_qurry"

dist_Road <- raster("./data/resampled/dist_Road.tif")
names(dist_Road) <- "dist_Road"

dist_Wasteplate <- raster("./data/resampled/dist_Wasteplate.tif")
names(dist_Wasteplate) <- "dist_Wasteplate"

Elevation <- raster("./data/resampled/Elevation.tif")
names(Elevation) <- "Elevation"

LST <- raster("./data/resampled/LST.tif")
names(LST) <- "LST"

Nightlit <- raster("./data/resampled/Nightlit.tif")
names(Nightlit) <- "Nightlit"

Rainfall <- raster("./data/resampled/Rainfall.tif")
names(Rainfall) <- "Rainfall"

SAVI <- raster("./data/resampled/SAVI.tif")
names(SAVI) <- "SAVI"

Slope <- raster("./data/resampled/Slope.tif")
names(Slope) <- "Slope"

Tempreature <- raster("./data/resampled/Tempreature.tif")
names(Tempreature) <- "Tempreature"

TRI <- raster("./data/resampled/TRI.tif")
names(TRI) <- "TRI"

TWI <- raster("./data/resampled/TWI.tif")
names(TWI) <- "TWI"

UI <- raster("./data/resampled/UI.tif")
names(UI) <- "UI"

WEX <- raster("./data/resampled/WEX.tif")
names(WEX) <- "WEX"

wind_speed <- raster("./data/resampled/wind_speed.tif")
names(wind_speed) <- "wind_speed"

# Stack the rasters
Rasters <- stack(Atmospher_density, Elevation, LST, Nightlit, Rainfall, Road_density,
                 SAVI, Slope, TRI, TWI, Tempreature, UI, WEX, buld_density,
                 dist_Airport, dist_Construction, dist_Petrolstation, dist_Railway,
                 dist_Road, dist_Unionmarket, dist_Wasteplate, dist_busstation,
                 dist_industry, dist_qurry, wind_speed)
plot(Rasters$Atmospher_density)
names(Rasters)

# Convert raster stack to a data frame with coordinates
Rasters.df <- as.data.frame(Rasters, xy = TRUE, na.rm = TRUE)
# Remove coordinate columns for normalization
Rasters.df_N <- Rasters.df[, -c(1, 2)]
Rasters.df_N_Nor <- as.data.frame(lapply(Rasters.df_N, normalize))
str(Rasters.df_N_Nor)

# ---- Produce Probability Maps ----
p3_prob <- as.data.frame(predict(fit.KNNAll, Rasters.df_N_Nor, type = "prob"))
summary(p3_prob)
Rasters.df$Levels_Moderate <- p3_prob$Moderate
Rasters.df$Levels_good     <- p3_prob$Good
Rasters.df$Levels_UHealSesn<- p3_prob$UHealSesn

# Create a spatial points data frame using coordinate columns
sp_points <- SpatialPointsDataFrame(Rasters.df[, c("x", "y")], data = Rasters.df)
r_ave_good <- rasterFromXYZ(as.data.frame(sp_points)[, c("x", "y", "Levels_good")])
proj4string(r_ave_good) <- CRS(projection(Elevation))
r_ave_Moderate <- rasterFromXYZ(as.data.frame(sp_points)[, c("x", "y", "Levels_Moderate")])
proj4string(r_ave_Moderate) <- CRS(projection(Elevation))
r_ave_UHealSesn <- rasterFromXYZ(as.data.frame(sp_points)[, c("x", "y", "Levels_UHealSesn")])
proj4string(r_ave_UHealSesn) <- CRS(projection(Elevation))

# Plot and export probability maps
spplot(r_ave_Moderate, main = "Moderate KNN Prediction")
writeRaster(r_ave_Moderate, filename = "Prediction_KNN_Tuned_Ave_Moderate.tif", 
            format = "GTiff", overwrite = TRUE)
spplot(r_ave_UHealSesn, main = "UHealSesn KNN Prediction")
writeRaster(r_ave_UHealSesn, filename = "Prediction_KNN_Tuned_Ave_UHealSesn.tif", 
            format = "GTiff", overwrite = TRUE)
spplot(r_ave_good, main = "Good KNN Prediction")
writeRaster(r_ave_good, filename = "Prediction_KNN_Tuned_Ave_GOOD.tif", 
            format = "GTiff", overwrite = TRUE)

# ---- Produce Classification Map ----
p3_class <- as.data.frame(predict(fit.KNNAll, Rasters.df_N_Nor, type = "raw"))
summary(p3_class)
# Add the predicted class to the data frame
Rasters.df$LevelAve <- p3_class[, 1]

# Import levels ID file (assumed to be in the repository root)
ID <- read.csv("./Levels_key.csv", header = TRUE)
grid_new <- join(Rasters.df, ID, by = "LevelAve", type = "inner")
grid_new_na <- na.omit(grid_new)
head(grid_new_na, 2)

# Convert to spatial points and then to a raster
sp_points_class <- SpatialPointsDataFrame(grid_new_na[, c("x", "y")], data = grid_new_na)
r_ave_class <- rasterFromXYZ(as.data.frame(sp_points_class)[, c("x", "y", "Level_ID")])
proj4string(r_ave_class) <- CRS(projection(Elevation))

# Export the final classification map
writeRaster(r_ave_class, filename = "Classification_Map_KNN_Tuned_Ave.tif", 
            format = "GTiff", overwrite = TRUE)

# Plot the landuse map with a custom color palette
myPalette <- colorRampPalette(c("light green", "#FFFF00", "#FFC600", "#FFAA00", "#FF3800", "#FF0000"))
LU_ave <- spplot(r_ave_class, "Level_ID", main = "PM10 Concentration Prediction: KNN", 
                 colorkey = list(space = "right", tick.number = 1, height = 1, width = 1.5,
                                 labels = list(at = seq(1, 4.8, length = 5), cex = 1.0,
                                               lab = c("Good", "Moderate", "UHealSesn", "UHeal", "Haz"))),
                 col.regions = myPalette, cut = 4)
print(LU_ave)
jpeg("Prediction_Map_KNN_AveAll.jpg", width = 1000, height = 700)
print(LU_ave)
dev.off()
