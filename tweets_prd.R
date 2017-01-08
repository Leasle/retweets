install.packages("tm")
install.packages("SnowballC")
install.packages("e1071")
install.packages("randomForest")
install.packages("caret")

library(tm)
library(SnowballC)
library(e1071)
library(caret)
library(randomForest)

train <- read.csv("train.csv", header = T)
test <- read.csv("test.csv", header = T)

train.part <- train[, c("id", "text", "in_reply_to_user_id", "user.description", 
                        "user.followers_count", "retweet_count")]

names(train.part) <- c("id", "text", "is_retweet", "description", "followers", "retweet_count")
train.part <- na.omit(train.part)


onlyText <- function(x) { 
  x <- gsub("[^a-zA-Z]", " ", x)
  return(gsub("http([[:alnum:]|[:blank:]]*)$", "", x)) 
}

tokenize <- function(x) {
  x <- tolower(x)
  x <- unlist(strsplit(x, split=" "))
  x
}


stopWords <- stopwords("en")
stopWords <- c(stopWords, "re", "ve", "ll", "dr")


removeStopWords <- sapply(1:10000, function(x) {
  t <- train.part[x, 2]
  t <- onlyText(t)
  t <- tokenize(t)
  t <- t[nchar(t) > 1]
  t <- t[!t %in% stopWords]
  
  paste(t, collapse=" ") 
})


train.part.vector <- VectorSource(removeStopWords) 
train.part.corpus <- Corpus(train.part.vector, 
                                      readerControl = list(language = "en"))
train.part.bag <- DocumentTermMatrix(train.part.corpus, 
                                control = list(stemming = TRUE))

train.part.bag <- removeSparseTerms(train.part.bag, 0.9988) 
dim(train.part.bag)

train.part[, 3] <- ifelse(train.part$is_retweet == "0.0" | train.part$is_retweet == "", 1, 0) 

train.df <- data.frame(inspect(train.part.bag[1:nrow(train.part.bag), ]))
is.retweeted <- ifelse(train.part$retweet_count[1:nrow(train.part.bag)] >= 20, 1, 0)
train.df <- cbind(is.retweeted, train.df)
train.df <- cbind(train.part[1:nrow(train.part.bag), 3], 
                  train.part[1:nrow(train.part.bag), 5], 
                  train.df)

vocab <- names(train.df)[-c(1,2,3)]

names(train.df)[1:2] <- c("is.retweet", "followers")

rm(train.part.bag)
rm(train.part.corpus)
rm(train.part.vector)
rm(removeStopWords)


t.start <- Sys.time()
set.seed(100)
forest <- train(as.factor(is.retweeted) ~ ., data = train.df,
                method = "rf",
                trControl = trainControl(method = "cv", number = 2),
                prox = TRUE,
                ntree = 20,
                do.trace = 10,
                allowParallel = TRUE)
t.end <- Sys.time()

t.end - t.start
print(forest)


test.part <- test[, c("id", "text", "in_reply_to_user_id", "user.description", 
                        "user.followers_count")]
names(test.part) <- c("id", "text", "is_retweet", "description", "followers")
test.part <- na.omit(test.part)
test.part[, 3] <- ifelse(test.part$is_retweet == "0.0" | test.part$is_retweet == "", 1, 0)


remove.stop.words.test <- sapply(1:nrow(test.part), function(x) {
  t <- test.part[x, 2]
  t <- onlyText(t)
  t <- tokenize(t)
  t <- t[nchar(t) > 1]
  t <- t[!t %in% stopWords]
  
  paste(t, collapse=" ") 
})


test.part.vector <- VectorSource(remove.stop.words.test) 
test.part.corpus <- Corpus(test.part.vector, 
                            readerControl = list(language = "en"))
test.part.bag <- DocumentTermMatrix(test.part.corpus, 
                                     control = list(stemming = TRUE, dictionary = vocab))

dim(test.part.bag)

test.df <- data.frame(inspect(test.part.bag[1:nrow(test.part.bag), ]))
test.df <- cbind(rep(2, nrow(test.df)), test.df)
test.df <- cbind(test.part[1:nrow(test.part.bag), 3], 
                       test.part[1:nrow(test.part.bag), 5], 
                       test.df)
names(test.df)[c(1,2)] <- c("is.retweet", "followers")


test.df[, 3] <- predict(forest, newdata = test.df, prob = T)

result.df <- data.frame(test.part$id, result = test.df[, 3])
names(result.df)[1] <- "id"
result.df[, 2] <- result.df[, 2] - 1
str(result.df)

write.csv(result.df, "prediction.csv", quote = FALSE, row.names = FALSE)
