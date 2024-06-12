setwd("C:/Users/35466/Desktop/Python Projects/BPNN_test")

data <- read.csv("output_pixel_data.csv")

head(data)
str(data)

# histogram of variable-1
hist(data$B02,col='red')

# histogram of variable-2
hist(data$B03,col='green',add=TRUE) 
# histogram of variable-3 
hist(data$B04,col='blue',add=TRUE)
# histogram of variable-3 
hist(data$B08,col='purple',add=TRUE)



cropped_data <- subset(data, Average_depth <= -0.7)


min(data$Average_depth)
max(data$Average_depth)

length(unique(data$Average_depth))

model <- lm(data$Average_depth ~ data$B02 + data$B03 + data$B04 + data$B08)
summary(model)

unique(data)
write.csv(unique(data), "dup_data.csv",, row.names = FALSE)

plot(data$B02 ~ data$Average_depth, xlab="Depth",ylab="Red")
abline(lm(data$B02 ~ data$Average_depth), col = "black")

plot(data$B04 ~ data$Average_depth, xlab="Depth",ylab="Blue")
abline(lm(data$B04 ~ data$Average_depth), col = "black")

plot(data$B03 ~ data$Average_depth, xlab="Depth",ylab="Green")
abline(lm(data$B03 ~ data$Average_depth), col = "black")

plot(data$B08 ~ data$Average_depth, xlab="Depth",ylab="Infra")
abline(lm(data$B08 ~ data$Average_depth), col = "black")

cor(data[ ,c("B02","B04","B03","B08","Average_depth")], use="pairwise.complete.obs")

qqnorm(data$Average_depth)
qqline(data$Average_depth)

qqnorm(data$B02)
qqline(data$B02)

qqnorm(data$B03)
qqline(data$B03)

qqnorm(data$B04)
qqline(data$B04)

qqnorm(data$B08)
qqline(data$B08)

#cropped data


cropped_model <- lm(cropped_data$Average_depth ~ cropped_data$B02 + cropped_data$B03 + cropped_data$B04 + cropped_data$B08)
summary(cropped_model)

qqnorm(cropped_data$Average_depth)
qqline(cropped_data$Average_depth)

qqnorm(cropped_data$B02)
qqline(cropped_data$B02)

qqnorm(cropped_data$B03)
qqline(cropped_data$B03)

qqnorm(cropped_data$B04)
qqline(cropped_data$B04)

qqnorm(cropped_data$B08)
qqline(cropped_data$B08)

plot(cropped_data$B02 ~ cropped_data$Average_depth, xlab="Depth",ylab="Red")
abline(lm(cropped_data$B02 ~ cropped_data$Average_depth), col = "black")

plot(cropped_data$B04 ~ cropped_data$Average_depth, xlab="Depth",ylab="Blue")
abline(lm(cropped_data$B04 ~ cropped_data$Average_depth), col = "black")

plot(cropped_data$B03 ~ cropped_data$Average_depth, xlab="Depth",ylab="Green")
abline(lm(cropped_data$B03 ~ cropped_data$Average_depth), col = "black")

plot(cropped_data$B08 ~ cropped_data$Average_depth, xlab="Depth",ylab="Infra")
abline(lm(cropped_data$B08 ~ cropped_data$Average_depth), col = "black")

cor(cropped_data[ ,c("B02","B04","B03","B08","Average_depth")], use="pairwise.complete.obs")

print(cov(data$Average_depth, data$B02, method = "pearson"))  

print(cov(data$Average_depth, data$B03, method = "pearson"))  

print(cov(data$Average_depth, data$B04, method = "pearson"))  

