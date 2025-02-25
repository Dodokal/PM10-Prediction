##### Data Preparation ----

# Load required R packages
library(caret)         # Machine learning
library(randomForest)  # Random Forest algorithm
library(rgdal)         # Spatial data processing
library(raster)        # Raster processing
library(plyr)          # Data manipulation
library(dplyr)         # Data manipulation
library(RStoolbox)     # Plotting spatial data
library(RColorBrewer)  # Color palettes
library(ggplot2)       # Plotting
library(sp)            # Spatial data
library(doParallel)    # Parallel processing

# -----------------------------
# Step 1: Import Data
# -----------------------------

original <- read.csv("./S2.csv", header = TRUE, stringsAsFactors = FALSE)
original <- na.omit(original)
original <- data.frame(original)  # Remove unwanted attributes

# -----------------------------
# Step 2: Data Cleaning
# -----------------------------
# Select only the required variables
original <- original[c("LevelAve", "Atmospher_density", "Elevation", "LST", "Nightlit",
                       "Rainfall", "Road_density", "SAVI", "Slope", "TRI", "TWI",
                       "Tempreature", "UI", "WEX", "buld_density", "dist_Airport",
                       "dist_Construction", "dist_Petrolstation", "dist_Railway",
                       "dist_Road", "dist_Unionmarket", "dist_Wasteplate", "dist_busstation",
                       "dist_industry", "dist_qurry", "wind_speed")]
str(original)
original$LevelAve <- as.factor(original$LevelAve)
summary(original)

# -----------------------------
# Step 3: Data Normalization
# -----------------------------
normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}
# Normalize all predictor columns (exclude the response "LevelAve")
original.n <- as.data.frame(lapply(original[, 2:26], normalize))
names(original.n)

# -----------------------------
# Step 4: Data Splitting
# -----------------------------
set.seed(123)
data.d <- sample(1:nrow(original), size = nrow(original) * 0.70, replace = FALSE)
train.data <- original.n[data.d, ]   # 70% training data
test.data  <- original.n[-data.d, ]    # Remaining 30% testing data

# Create target variable data frames
train.data_labels <- original[data.d, 1]
test.data_labels  <- original[-data.d, 1]

train.data$PM <- train.data_labels
original.n$PM <- original$LevelAve

# -----------------------------
# Step 5: Model Training & Tuning
# -----------------------------

# (a) Default Random Forest settings
trControl <- trainControl(method = 'repeatedcv', 
                          number = 10, 
                          repeats = 3, 
                          search = "grid")

set.seed(1234)
rf_defaultN <- train(PM ~ Atmospher_density + Elevation + LST + Nightlit + Rainfall +
                       Road_density + SAVI + Slope + TRI + TWI + Tempreature + UI +
                       WEX + buld_density + dist_Airport + dist_Construction +
                       dist_Petrolstation + dist_Railway + dist_Road + dist_Unionmarket +
                       dist_Wasteplate + dist_busstation + dist_industry + dist_qurry +
                       wind_speed,
                     data = train.data,
                     method = "rf",
                     metric = "Accuracy",
                     trControl = trControl)
print(rf_defaultN)
plot(rf_defaultN)
p1_defaultN <- predict(rf_defaultN, test.data, type = "raw")
confusionMatrix(p1_defaultN, as.factor(test.data_labels))

# (b) Tuning mtry parameter
set.seed(1234)
tuneGrid <- expand.grid(.mtry = 1:20)
rf_mtry <- train(PM ~ SAVI + UI + Slope + Elevation + LST + Rainfall + Tempreature +
                   Nightlit + wind_speed + dist_Airport + dist_Construction +
                   dist_Petrolstation + dist_Railway + dist_Road + dist_Unionmarket +
                   dist_Wasteplate + dist_busstation + dist_industry + dist_qurry,
                 data = train.data,
                 method = "rf",
                 metric = "Accuracy",
                 tuneGrid = tuneGrid,
                 trControl = trControl,
                 importance = TRUE)
print(rf_mtry)
best_mtry <- rf_mtry$bestTune$mtry
print(best_mtry)

# (c) Searching for best maxnodes (optional)
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in 5:30) {
  set.seed(1234)
  rf_maxnode <- train(PM ~ SAVI + UI + Slope + Elevation + LST + Rainfall +
                        Tempreature + Nightlit + wind_speed + dist_Airport +
                        dist_Construction + dist_Petrolstation + dist_Railway +
                        dist_Road + dist_Unionmarket + dist_Wasteplate +
                        dist_busstation + dist_industry + dist_qurry,
                      data = train.data,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = maxnodes,
                      ntree = 300)
  store_maxnode[[as.character(maxnodes)]] <- rf_maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)
results_mtry

# (d) Final model training with tuned parameters
fit_rf_final <- train(PM ~ Atmospher_density + Elevation + LST + Nightlit + Rainfall +
                        Road_density + SAVI + Slope + TRI + TWI + Tempreature + UI +
                        WEX + buld_density + dist_Airport + dist_Construction +
                        dist_Petrolstation + dist_Railway + dist_Road + dist_Unionmarket +
                        dist_Wasteplate + dist_busstation + dist_industry + dist_qurry +
                        wind_speed,
                      data = train.data,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE)
print(fit_rf_final)
varImp(fit_rf_final)
plot(varImp(fit_rf_final), main = "RF Tuned Model")

# Evaluate final model
p1_final <- predict(fit_rf_final, test.data, type = "raw")
confusionMatrix(p1_final, as.factor(test.data_labels))

# (e) Random search tuning for comparison
control_random <- trainControl(method = 'repeatedcv', 
                               number = 10, 
                               repeats = 3, 
                               search = 'random')
set.seed(1)
rf_random <- train(PM ~ Atmospher_density + Elevation + LST + Nightlit + Rainfall +
                     Road_density + SAVI + Slope + TRI + TWI + Tempreature + UI +
                     WEX + buld_density + dist_Airport + dist_Construction +
                     dist_Petrolstation + dist_Railway + dist_Road + dist_Unionmarket +
                     dist_Wasteplate + dist_busstation + dist_industry + dist_qurry +
                     wind_speed,
                   data = train.data,
                   method = 'rf',
                   metric = 'Accuracy',
                   trControl = control_random,
                   importance = TRUE)
print(rf_random)
varImp(rf_random)
plot(varImp(rf_random))
plot(rf_random)
p1_random <- predict(rf_random, test.data, type = "raw")
confusionMatrix(p1_random, as.factor(test.data_labels))

# Save variable importance plot
jpeg("varImportance_RF.jpg", width = 800, height = 500)
plot(varImp(fit_rf_final), main = "Variable Importance - RF Tuned Model")
dev.off()

# -----------------------------
# Step 6: ROC Curves for Model Evaluation
# -----------------------------
library(pROC)
predictions <- as.data.frame(predict(rf_random, test.data, type = "prob"))
# Determine predicted class based on highest probability
predictions$predict <- colnames(predictions)[apply(predictions[, 1:3], 1, which.max)]
predictions$observed <- test.data_labels
head(predictions)

roc_Good <- roc(ifelse(predictions$observed == "Good", "Good", "non-Good"), as.numeric(predictions$Good))
roc_Moderate <- roc(ifelse(predictions$observed == "Moderate", "Moderate", "non-Moderate"), as.numeric(predictions$Moderate))
roc_UHealSesn <- roc(ifelse(predictions$observed == "UHealSesn", "UHealSesn", "non-UHealSesn"), as.numeric(predictions$UHealSesn))

plot(roc_Moderate, col = "blue", main = "RF Tuned Model ROC Curve", xlim = c(0.64, 0.1))
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
# Step 7: Train Model Using All Data
# -----------------------------
original.n$PM <- original$LevelAve
set.seed(849)
fit_rfAll <- train(PM ~ Atmospher_density + Elevation + LST + Nightlit + Rainfall +
                     Road_density + SAVI + Slope + TRI + TWI + Tempreature + UI +
                     WEX + buld_density + dist_Airport + dist_Construction +
                     dist_Petrolstation + dist_Railway + dist_Road + dist_Unionmarket +
                     dist_Wasteplate + dist_busstation + dist_industry + dist_qurry +
                     wind_speed,
                   data = original.n,
                   method = "rf",
                   metric = "Accuracy",
                   tuneGrid = tuneGrid,
                   trControl = trControl,
                   importance = TRUE)
X_rfAll <- varImp(fit_rfAll)
plot(X_rfAll)
print(fit_rfAll$results)
jpeg("varImportance_All_RF.jpg", width = 800, height = 500)
plot(X_rfAll, main = "Variable Importance - All Data RF")
dev.off()

# -----------------------------
# Step 8: Produce Prediction Map Using Raster Data
# -----------------------------
# Assuming raster files are placed in a "data/resampled" folder within the repository

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
plot(Rasters$SAVI)
names(Rasters)

# Convert raster stack to data frame with coordinates
Rasters.df <- as.data.frame(Rasters, xy = TRUE, na.rm = TRUE)
coor <- Rasters.df %>% select(x, y)
Rasters.df_N <- Rasters.df[, !(names(Rasters.df) %in% c("x", "y"))]

# Normalize the raster data
Rasters.df_N_Nor <- as.data.frame(lapply(Rasters.df_N, normalize))
str(Rasters.df_N_Nor)

# Produce probability maps using the all-data model (fit_rfAll)
p3 <- as.data.frame(predict(fit_rfAll, Rasters.df_N_Nor, type = "prob"))
summary(p3)
Rasters.df$Levels_UHealSesn <- p3$UHealSesn
Rasters.df$Levels_good      <- p3$Good
Rasters.df$Levels_Moderate  <- p3$Moderate

x_sp <- SpatialPointsDataFrame(Rasters.df[, c("x", "y")], data = Rasters.df)
r_ave_good <- rasterFromXYZ(as.data.frame(x_sp)[, c("x", "y", "Levels_good")])
proj4string(r_ave_good) <- CRS(projection(Elevation))

r_ave_Moderate <- rasterFromXYZ(as.data.frame(x_sp)[, c("x", "y", "Levels_Moderate")])
proj4string(r_ave_Moderate) <- CRS(projection(Elevation))

r_ave_UHealSesn <- rasterFromXYZ(as.data.frame(x_sp)[, c("x", "y", "Levels_UHealSesn")])
proj4string(r_ave_UHealSesn) <- CRS(projection(Elevation))

# Plot and save prediction maps
spplot(r_ave_Moderate, main = "Moderate PM10 Concentration")
writeRaster(r_ave_Moderate, filename = "Prediction_RF_Tuned_Ave_Moderate.tif", 
            format = "GTiff", overwrite = TRUE)

spplot(r_ave_UHealSesn, main = "UHealSesn PM10 Concentration")
writeRaster(r_ave_UHealSesn, filename = "Prediction_RF_Tuned_Ave_UHealSesn.tif", 
            format = "GTiff", overwrite = TRUE)

spplot(r_ave_good, main = "Good PM10 Concentration")
writeRaster(r_ave_good, filename = "Prediction_RF_Tuned_Ave_GOOD.tif", 
            format = "GTiff", overwrite = TRUE)

# -----------------------------
# Step 9: Produce Classification Map
# -----------------------------
p3_class <- as.data.frame(predict(fit_rfAll, Rasters.df_N_Nor, type = "raw"))
summary(p3_class)
Rasters.df$Levels_ave <- p3_class[, 1]
head(Rasters.df, 2)

# Import Levels ID file (assumed to be in the repository root)
ID <- read.csv("./Levels_key.csv", header = TRUE)
grid_new <- join(Rasters.df, ID, by = "Levels_ave", type = "inner")
grid_new_na <- na.omit(grid_new)
head(grid_new_na, 2)

x_sp_class <- SpatialPointsDataFrame(grid_new_na[, c("x", "y")], data = grid_new_na)
r_ave <- rasterFromXYZ(as.data.frame(x_sp_class)[, c("x", "y", "Level_ID")])
proj4string(r_ave) <- CRS(projection(Elevation))

# Export final classification map
writeRaster(r_ave, filename = "Classification_Map_RF_Tuned_Ave.tif", 
            format = "GTiff", overwrite = TRUE)

# Plot Landuse Map with custom color palette
myPalette <- colorRampPalette(c("light green", "#FFFF00", "#FFC600", "#FFAA00", "pink", "red"))
LU_ave <- spplot(r_ave, "Level_ID", main = "PM10 Ave. Concentration Prediction: RF", 
                 colorkey = list(space = "right", tick.number = 1, height = 1, width = 1.5,
                                 labels = list(at = seq(1.2, 5.9, length = 6), cex = 1.0,
                                               lab = c("Good", "Moderate", "UHealSesn", "UHeal", "VUnheal", "UHealSesn"))),
                 col.regions = myPalette, cut = 5)
print(LU_ave)
jpeg("Prediction_Map_RF_Tuned_AveAllRec.jpg", width = 1000, height = 700)
print(LU_ave)
dev.off()

print("############################# Output Ends here and follow######################")
