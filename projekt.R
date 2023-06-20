# ZBIERANIE DANYCH
# Źródło: https://www.kaggle.com/datasets/camnugent/california-housing-prices
# wczytywanie danych
data <- read.csv("/Users/aniamarjankowska/Documents/studies/II\ DEGREE/VIII/WdAD/project/housing.csv", stringsAsFactors = FALSE)


# EKSPLORACJA I PRZYGOTOWANIE DANYCH
# sprawdzenie struktury i poprawności danych
# oraz detekcja wartosci NaN
str(data)
summary(data)
sapply(data, function(x) sum(is.na(x)))

# zauważamy NaNy w kolumnie total_bedrooms, ale w porównaniu do wielkości próby
# jest ich niewiele, zatem decydujemy się na usunięcie wierszy z NaNami
nrow(data)
data <- na.omit(data)
nrow(data)

# usuwamy cechy latitude and longitude, ponieważ nie mają one 
# wpływu na cenę mieszkań
data1 <- subset(data, select = -c(longitude, latitude))

# one-hot encoding dla zmiennej kategorycznej ocean_proximity
library(mltools)
library(data.table)
unique(data$ocean_proximity)
data1$ocean_proximity <- as.factor(data1$ocean_proximity)
data1 <- one_hot(as.data.table(data1))
View(data1)

# macierz korelacji, heatmapa
res <- cor(data1)
round(res, 2)
heatmap(x = res, symm = TRUE)

# mieszanie danych
shuffled_data <- data1[sample(1:nrow(data1)),]
summary(shuffled_data$median_house_value)

# histogram zmiennej wyjściowej
hist(shuffled_data$median_house_value, breaks = "Sturges", xlab = "Median House Value", 
     main = "Histogram zmiennej wyjściowej", xlim = range(0, 580000))

# detekcja outlierów metodą Z-score
zscore_cols <- names(shuffled_data)[-(7:12)]
shuffled_data <- data.frame(shuffled_data)
for(col in zscore_cols){
  z_scores <- (shuffled_data[,col] - mean(shuffled_data[,col])) / sd(shuffled_data[,col])
  print(paste(col, nrow(shuffled_data[which(z_scores > 2 | z_scores < -2),])))
}

# box ploty zmiennych objaśniających
# oraz detekcja outlierów za pomocą metody IQR
box_cols <- names(shuffled_data)[-(7:12)]
View(shuffled_data)
for(col in box_cols){
  boxplot(shuffled_data[,col], main = paste("Box plot zmiennej", col))
  iqr <- unname(quantile(shuffled_data[,col], .75) - quantile(shuffled_data[,col], .25))
  ul <- unname(quantile(shuffled_data[,col], .75) + 1.5 * iqr)
  ll <- unname(quantile(shuffled_data[,col], .25) - 1.5 * iqr)
  print(paste(col, nrow(shuffled_data[which(shuffled_data[,col] > ul | shuffled_data[,col] < ll),])))
}

# widzimy, że jest dużo outlieróww jeżeli rozdzielimy dane na dane jednowymiarowe, 
# musimy to sprawdzić. Użyjemy więc odległości Mahalanobisa do wykrywania takowych,
# ponieważ mamy tutaj do czynienia z danymi wielowymiarowymi
x_data <- shuffled_data[, -c(7:12)]

# wyznaczanie ogległości Mahalonobisa dla każdej z obserwacji
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

# zakładamy, że 10% obserwacji to wartości odstające
distances > quantile(distances, .9)
x_data[distances <= quantile(distances, .9),]
nrow(x_data[distances <= quantile(distances, .9),])

row(shuffled_data)

# skalowanie zmiennych w naszym projekcie nie jest potrzebne,
# ponieważ używamy wyłącznie modeli opartych na 
# strukturach drzewiastych, które nie wymagają w takiej formie
# przeskalowanych zmiennych, gdyż nie wykorzystują w żadnym sensie
# pojęcia metryki

# podział danych
# 80% - train, 20% - test
data_train <- shuffled_data[1:16512, ]
data_test <- shuffled_data[16513:20433, ]


# BUDOWA MODELU
# drzewo regresyjne
library(rpart)
m.rpart <- rpart(median_house_value~., data = data_train)
p.rpart <- predict(m.rpart, data_test)

summary(p.rpart)

# implementacja trzech metryk do oceny modelu
# RMSE - root mean squared error
# MAE - mean absolute error
# MSE - mean squared error
RMSE <- function(actual, predicted){
  sqrt(mean((actual - predicted) ** 2))
}

MAE <- function(actual, predicted){ 
  mean(abs(actual-predicted))
}

MSE <- function(actual, predicted){ 
  mean((actual-predicted)**2)
}

# wyznaczenie współczynnika korelacji między prognozami modelu
# a zmienną objaśnianą ze zbioru testowego
# oraz wartości metryk zaimplementowanych powyżej
cor(p.rpart, data_test$median_house_value)

MAE(p.rpart, data_test$median_house_value)

RMSE(p.rpart, data_test$median_house_value)

MSE(p.rpart, data_test$median_house_value)

# wyznaczenie MAE dla wartości objaśnianej w zbiorze testowym oraz
# średniej wartości zmiennej objaśnianej ze zbioru testowego

RMSE(mean(data_train$median_house_value), data_test$median_house_value)
# model zachowuje się lepiej niż modelowanie wartości objaśnianej 
# wartością średnią ze zmiennej wyjściowej

# wizualizacja drzewa regresyjnego
library(rpart.plot)
rpart.plot(m.rpart, digits = 3)


# DOPRACOWANIE MODELU
# drzewo modeli
library(Cubist)

# model
m.cubist <- cubist(data_train[-7], data_train$median_house_value)

# predykcje
p.cubist <- predict(m.cubist, data_test)

# wyznaczenie współczynnika korelacji między prognozami modelu
# a zmienną objaśnianą ze zbioru testowego
cor(p.cubist, data_test$median_house_value)

# wyznaczenie wartości metryki RMSE dla prognoz modelu
# oraz zmienną objaśnianą ze zbioru testowego
RMSE(data_test$median_house_value, p.cubist)

# podstawowe statystyki prognoz
m.cubist
summary(m.cubist)

# najmniejszy błąd na zbiorze testowym otrzymujemy dla drzewa modeli
