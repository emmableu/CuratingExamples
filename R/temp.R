## view the first few rows of the data mydata <-
# library(ridge)
library(dplyr)
mydata <- read.csv("pattern_df_train3.csv")
# mydata <- read.csv("temp_train.csv")
head(mydata)
lambda_seq <- 10^seq(2, -2, by = -.1)

fit <- lm(zzzY ~ ., data=mydata[, 2:129])

# fit <- lm(zzzY ~ createClone+ receiveOnClone + removeClone, data=mydata[,2:129])
# fit <- lm(y ~ . , data = mydata[2:5])
summary(fit)
cor_data = (cor(mydata))
View(cor_data[,c('reportMouseX')])
