#Installing the required package for extracting the audio files
library(tuneR)

#Path of the data from where the audio should be fetched
audiopath <- ("C:/Users/Nithiarashu/Desktop/Thesis/Musical Instrument/Audio")

#Get all the files names in a vector
fnames <- as.vector(list.files(path = audiopath, full.names = TRUE, include.dirs = FALSE))

audiodata <- c()
features <- c()

#Read all the audio files

for (i in 1:length(fnames)) {audiodata <- readMP3(fnames[[i]])

#Extracting the Mel-frequency cepstral coefficients (MFCC)

mfeature <- melfcc(audiodata,44100, wintime=0.016, lifterexp=0, numcep = 12, sumpower = FALSE, 
             nbands = 40, bwidth = 1, minfreq=133.33,maxfreq=22050,fbtype = "mel")

#Getting the MFCC features of the audio files in a vector 

mfccfeature <- as.vector(mfeature)

#Getting the Zero Crossing Rate Features from the audio files in a Vector 

library(seewave)
Zcrfeature <- data.frame(zcr(audiodata,w1=NULL))
#str(Zcrfeature)
zerocrossingrate <- as.vector(Zcrfeature$zcr)

#Combaining MFCC and ZCR features

combainedfeatures <- c(zerocrossingrate,mfccfeature)

cfeatures <- as.vector(combainedfeatures)

features[[i]] <- cfeatures

}

#Installing libraries for sepctrogram plot 

library(sound)
library(oce)
library(signal, warn.conflicts = F, quietly = T)
library(oce, warn.conflicts = F, quietly = T)

# number of points to use for the fft
nfft=1024
# window size (in points)
window=256
# overlap (in points)
overlap=128
# sr
sr=48000
o <- audiodata
str(o)
#create spectrogram
spec <- specgram(x = audiodata@left, n = nfft, Fs=sr, window = window, overlap = overlap)
str(spec)
# discard phase information
P = abs(spec$S)
# normalize
P = P/max(P)
# convert to dB
P = 10*log10(P)
# config time axis
t = spec$t
# Spectrogram Plot
spectrogramplot <- imagep(x = t,
                y = spec$f,
                z = t(P),
                ylab = 'Frequency [Hz]',
                xlab = 'Time [s]',
                drawPalette = T,
                decimate = F,
                main='Spectrogram Plot',
                col=c("lightgray", "navy")
)

#Converting the vector into data frame

dframe <- as.data.frame(do.call(rbind,features))
    
#Read the labels of the audio files

audiolabels <- read.csv(file = "C:/Users/Nithiarashu/Desktop/Thesis/Musical Instrument/Labels.csv", header = T)

#Merge the labels to the data frame we have created earlier

dframe1 <- data.frame(cbind(audiolabels$Category, dframe))
names(dframe1)[1] <- c("label")
table(dframe1$label)
dframe1$label <- factor(dframe1$label)

#Finding the missing values

is.na(dframe1)
dframe1 <- na.omit(dframe1)

#Class Imbalance plot

plot(dframe1$label , xlab = 'Class',ylab = 'Count of audio', lwd = 2, col="cyan",main = "Class Imbalance")

#To make the dataset balanced
library(UBL)
dframe1 <- SmoteClassif(label ~ ., dframe1, C.perc = "balance", repl = FALSE)
dframe1 <- dframe1[1:15000]

#Class balance plot

plot(dframe1$label , xlab = 'Class',ylab = 'Count of audio', lwd = 1, col="cyan",main = "Class Imbalance")
table(dframe1$label)

#Feature Selection using Boruta

library(Boruta)
set.seed(17154154)
BorutaFeatures <- Boruta(label ~ .,data=dframe1,doTrace=2)
ExtractedFeatures <- (BorutaFeatures$finalDecision  =="Confirmed")
Bfeatures <- dframe1[,ExtractedFeatures]

#Boruta Tentative RoughFix 
BorutaFeatures_n <- TentativeRoughFix(BorutaFeatures)
ExtractedFeatures_n <- (BorutaFeatures_n$finalDecision =="Confirmed")
Bfeatures_n <- dframe1[,ExtractedFeatures_n]

#Building models 

library(caret)
library(e1071)
library(Metrics)
library(ggplot2)
library(ggthemes)
library(dplyr)
library(MLmetrics)

#Train and test split for classifiers
set.seed(17154154)
partition <- createDataPartition(dframe1$label, p=0.80, list = FALSE)
train <- Bfeatures_n[partition,]
test <- Bfeatures_n[-partition,]

#Random Forest Implementation 

library(randomForest)
forest <- randomForest(label ~ ., data=train,importance=TRUE, ntree=50, set.seed(17154154))
Rfpredict <- predict(forest, test[,-1])
(RForestAccuracy <- 1 - mean(Rfpredict != test$label))

#Confusion Matrix, Precision and Recall for Random Forest
caret::confusionMatrix(Rfpredict, test$label)
Rf <- table(Rfpredict, test$label) 
precisionRf <- diag(Rf)/rowSums(Rf)
RecallRf <- diag(Rf)/ colSums(Rf)

#AdaBoost Implementation

library(adabag)
Adaboost <- boosting(label~., data = train,  mfinal = 20,
                     coeflearn = "Breiman")
AboostPrediction <- predict.boosting(Adaboost, newdata = test[,c(-1)])
(AboostAccuracy <- 1 - mean(AboostPrediction$class != test$label))
AboostPrediction$class <- factor(AboostPrediction$class, levels = c(1:5), labels = c("Bassoon", "Doublebass", "French Horn", "trombone ", "viola")) 

#Confusion Matrix, Precision and Recall for AdaBoost
Adaboostconfusion <- table(Prediction= AboostPrediction$class,Reference= test$label)
precisionAb <- diag(Adaboostconfusion)/rowSums(Adaboostconfusion)
RecallAb <- diag(Adaboostconfusion)/ colSums(Adaboostconfusion)

#XGBoost Implementation

library(xgboost)
library(magrittr)
library(dplyr)
library(Matrix)

# Create train and test matrix for XGBoost
trainxg <- sparse.model.matrix(label~ .-1, data = train)
train_label <- train[,"label"]
train_matrix <- xgb.DMatrix(data = as.matrix(trainxg), label = train_label)

testxg <- sparse.model.matrix(label~ .-1, data = test)
test_label <- test[,"label"]
test_matrix <- xgb.DMatrix(data = as.matrix(testxg), label = test_label)

# Parameters for XGBoost

xgb_params <- list("objective" = "multi:softmax",
                   "eval_metric" = "mlogloss",
                   "num_class" = 6)
watchlist <- list(train = train_matrix, test = test_matrix)

# eXtreme Gradient Boosting Model
XGboost <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = 100,
                       watchlist = watchlist,
                       eta = 0.5,
                       max.depth = 3,
                       gamma = 0,
                       subsample = 1,
                       colsample_bytree = 1,
                       seed = 154)

# Training & testing log loss plot
mlogerror <- data.frame(XGboost$evaluation_log)
plot(mlogerror$iter, mlogerror$train_mlogloss, col = 'blue')
lines(mlogerror$iter, mlogerror$test_mlogloss, col = 'red')
legend(80,1.1,legend = c("Test","Train"),col=c("red","blue"),lty=1:2,cex=0.8)

# Prediction 
Xgprediction <- predict(XGboost, newdata = test_matrix)
Xgprediction <-factor(Xgprediction, levels = c(1:5), labels = c("Bassoon", "Doublebass", "French Horn", "trombone ", "viola"))

# plot only the first tree and display the node ID:
xgb.plot.tree(model = XGboost, trees = 0:1, show_node_id = TRUE)

#Confusion Matrix, Precision and Recall for XGBoost
caret::confusionMatrix(Xgprediction, test_label)
xg <-table(prediction = Xgprediction, Actual = test_label)
precisionxg <- diag(xg)/rowSums(xg)
Recallxg <- diag(xg)/ colSums(xg)




