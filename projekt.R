# ZBIERANIE DANYCH
# źrodło: https://www.kaggle.com/datasets/camnugent/california-housing-prices
# wczytywanie danych
data <- read.csv("/Users/weronikasadzik/Downloads/housing.csv", stringsAsFactors = FALSE)

# EKSPLORACJA I PRZYGOTOWANIE DANYCH
# sprawdzenie poprawności danych
str(data)
summary(data)
sum(is.null(data))
# drop latitude and longitude
data1 <- subset(data, select=-c(longitude,latitude))
# mieszanie danych
shuffled_data <- data1[sample(1:nrow(data1)),]
# histogram zmiennej wyjściowej
hist(shuffled_data$median_house_value, main="Histogram zmiennej wyjściowej")
# TODO: tu coś jeszcze do eksploracji danych by się przydało
# 80% - train, 20% - test
data_train <- shuffled_data[1:16512, ]
data_test <- shuffled_data[16513:20640, ]
  
# BUDOWA MODELU
# drzewo regresyjne
library(rpart)
m.rpart <- rpart(median_house_value~., data=data_train)
p.rpart <- predict(m.rpart, data_test)

# TODO: może zaimplementujemy MSE?
MAE <- function(actual, predicted){
  mean(abs(actual - predicted))
}

MAE(p.rpart, data_test$median_house_value)
# dużo!

MAE(mean(data_train$median_house_value), data_test$median_house_value)
# ale lepiej niż branie stale średniej

# DOPRACOWANIE MODELU
# TODO: ensemble learning- drzewo modeli
library(Cubist)