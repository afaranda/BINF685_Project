library(dplyr)
library(ggplot2)


#df<-data.frame(
#	Score = rnorm(1250,rep(seq(700, 680, by=-20/250), times=5), 5),
#	Iteration = rep(1:250, times=5),
#	Trial = as.factor(rep(1:5, each = 250))
#)



df<-read.table("Train_Results.txt", header=T, sep="\t")
png("Training_scores.png")
ggplot(data = df, aes(x=Iteration, y=Score)) + geom_line(aes(color=as.factor(Trial)))
dev.off()