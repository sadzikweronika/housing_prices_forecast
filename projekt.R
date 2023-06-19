# ZBIERANIE DANYCH
# źrodło: https://www.kaggle.com/datasets/camnugent/california-housing-prices
# wczytywanie danych
data <- read.csv("/Users/weronikasadzik/Downloads/housing.csv", stringsAsFactors = FALSE)


# EKSPLORACJA I PRZYGOTOWANIE DANYCH
# sprawdzenie poprawności danych
str(data)
summary(data)
sapply(data, function(x) sum(is.na(x)))

# zauważamy NaNy w kolumnie total_bedrooms, ale w porównaniu do wielkości próby
# jest ich niewiele, zatem decydujemy się na usunięcie wierszy z NaNami
nrow(data)
data <- na.omit(data)
nrow(data)

# drop latitude and longitude
data1 <- subset(data, select = -c(longitude, latitude))

# one-hot encoding dla zmiennej kategorycznej
library(mltools)
library(data.table)
unique(data$ocean_proximity)
data1$ocean_proximity <- as.factor(data1$ocean_proximity)
data1 <- one_hot(as.data.table(data1))
View(data1)

# mieszanie danych
shuffled_data <- data1[sample(1:nrow(data1)),]

# histogram zmiennej wyjściowej
hist(shuffled_data$median_house_value, main = "Histogram zmiennej wyjściowej")

# detekcja outlierów metodą Z-score
zscore_cols <- names(shuffled_data)[-(7:12)]
for(col in zscore_cols){
  z_scores <- (shuffled_data[,col] - mean(shuffled_data[,col])) / sd(shuffled_data[,col])
  print(paste(col, nrow(shuffled_data[which(z_scores > 2 | z_scores < -2),])))
}

# box ploty zmiennych objaśniających
# oraz detekcja outlierów za pomocą metody IQR
box_cols <- names(shuffled_data)[-(7:12)]
View(shuffled_data)
shuffled_data <- data.frame(shuffled_data)
for(col in box_cols){
  boxplot(shuffled_data[,col], main = paste("Box plot zmiennej", col))
  iqr <- unname(quantile(shuffled_data[,col], .75) - quantile(shuffled_data[,col], .25))
  ul <- unname(quantile(shuffled_data[,col], .75) + 1.5 * iqr)
  ll <- unname(quantile(shuffled_data[,col], .25) - 1.5 * iqr)
  print(paste(col, nrow(shuffled_data[which(shuffled_data[,col] > ul | shuffled_data[,col] < ll),])))
}

# widzimy, że jest dużo outlierów jeżeli rozdzielimy dane na dane jednowymiarowe, 
# musimy to sprawdzić, ale użyjemy odległości Mahalanobisa do wykrywania takowych,
# ponieważ mamy tutaj do czynienia z danymi wielowymiarowymi
x_data <- shuffled_data[, -7]
x_data <- shuffled_data[, -c(7:12)]
centroid <- unname(sapply(x_data,FUN=mean))
mahalonobis_dist <- t(matrix(as.numeric()))
distances <- c()
centroid
for (x in 1:20433) {
  example_distance <- x_data[x,] - centroid
  cov_mat <- cov(x_data)
  inv_cov_mat <- solve(cov_mat)
  mahalonobis_dist <- t(matrix(as.numeric(example_distance))) %*% 
    matrix(inv_cov_mat, nrow = 6) %*% matrix(as.numeric(example_distance))
  distances <- c(distances, mahalonobis_dist)
}
distances
distances > quantile(distances, .9)
x_data[distances <= quantile(distances, .9),]
nrow(x_data[distances <= quantile(distances, .9),])

# 80% - train, 20% - test
data_train <- shuffled_data[1:16512, ]
data_test <- shuffled_data[16513:20640, ]
warnings()


# BUDOWA MODELU
# drzewo regresyjne
library(rpart)
m.rpart <- rpart(median_house_value~., data=data_train)
p.rpart <- predict(m.rpart, data_test)

# TODO: może zaimplementujemy MSE?
RMSE <- function(actual, predicted){
  sqrt(mean((actual - predicted) ** 2))
}

RMSE(p.rpart, data_test$median_house_value)
# dużo!

RMSE(mean(data_train$median_house_value), data_test$median_house_value)
# ale lepiej niż branie stale średniej

# DOPRACOWANIE MODELU
# TODO: ensemble learning- drzewo modeli
library(Cubist)
m.cubist <- cubist(data_train[-7], data_train$median_house_value)
p.cubist <- predict(m.cubist, data_test)
RMSE(data_test$median_house_value, p.cubist)
