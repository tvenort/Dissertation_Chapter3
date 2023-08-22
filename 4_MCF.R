library(tidyverse)
library(ggplot2)
#Read model inputs
sample<-read.csv(paste('./8_Analyses/ML_GSA/gsa_sample.csv', sep=""))
head(sample)
dim(sample)
colnames(sample)
#Read model outputs on which you focus your interventions on 
outputs<- read.csv(paste('./8_Analyses/ML_GSA/gsa_runs0.csv', sep=""))
outputs
#read model final output
#output_final<- read.csv(paste('./8_Analyses/ML_models/prob_outputs_final.csv', sep=""))
#output_final
#combining two datasets
data<-cbind(sample,outputs)
head(data)
#View(data)

#----- K_PNB MCF--------------------------------------------
#distribution of behavioral vs non-behavioral subsets
min(data$K.PNB)
max(data$K.PNB)
ggplot()+
  geom_density(aes(K.PNB),alpha = 0.3, fill = "blue", data = data%>%filter(X0 < 0.33))+
  #geom_histogram(aes(MutMxSz),alpha = 0.3, fill = "blue",bins = 50, data = data%>%filter(AvgMutPop > 70))+
  geom_density(aes(K.PNB),alpha = 0.3, fill = "red", data = data%>%filter(X0 > 0.33))+
  geom_vline(xintercept = -7)
ggplot()+
  geom_histogram(aes(K.PNB),alpha = 0.3, fill = "blue", data = data%>%filter(X0 < 0.33))+
  geom_histogram(aes(K.PNB),alpha = 0.3, fill = "red", data = data%>%filter(X0 > 0.33))

breaks_low = seq(min(data$K.PNB[data$X0 < 0.33]),max(data$K.PNB[data$X0< 0.33]), 
                 by=abs(max((data$K.PNB[data$X0 < 0.33])/100)))
#max((data$K.PNB[data$Class_O > 0.5]))/100
data.cut = cut(data$K.PNB[data$X0 < 0.33], breaks_low, right=FALSE)
data.freq = table(data.cut)
data.cumfreq = cumsum(data.freq)
freq_low <- cbind(data.cumfreq) 
dim(freq_low)
freq_l <- freq_low/nrow(data%>%filter(X0 < 0.33))
dim(freq_l)

breaks_high = seq(min(data$K.PNB[data$X0 > 0.33]), 
                  max(data$K.PNB[data$X0 > 0.33]),
                  by=abs(max(data$K.PNB[data$X0 > 0.33])/100))
data.cut = cut(data$K.PNB[data$X0 > 0.33], breaks_high, right=FALSE)
data.freq = table(data.cut)
data.cumfreq = cumsum(data.freq)
freq_high <- cbind(data.cumfreq)
dim(freq_high)
freq_h <- freq_high/nrow(data%>%filter(X0 > 0.33))
dim(freq_h)

ggplot()+
  geom_line(aes(x = c(1:190), y = freq_h), colour = "red")+
  geom_line(aes(x = c(1:190), y = freq_l), colour = "blue")

ks.test(freq_l, freq_h, alternative = "two.sided")

#----- P_PNB MCF--------------------------------------------
ggplot()+
  
  geom_density(aes(P.PNB),alpha = 0.3, fill = "blue", data = data%>%filter(X0 < 0.33))+
  #geom_histogram(aes(MutMxSz),alpha = 0.3, fill = "blue",bins = 50, data = data%>%filter(AvgMutPop > 70))+
  geom_density(aes(P.PNB),alpha = 0.3, fill = "red", data = data%>%filter(X0 > 0.33))+
  geom_vline(xintercept = -6)
min(data$P.PNB)
max(data$P.PNB)
breaks_low = seq(min(data$P.PNB[data$X0< 0.33]),max(data$P.PNB[data$X0 < 0.33]), 
                 by=abs(max((data$P.PNB[data$X0<= 0.33])/100)))
#max((data$K.PNB[data$Class_O > 0.5]))/100
data.cut = cut(data$K.PNB[data$X0 < 0.33], breaks_low, right=FALSE)
data.freq = table(data.cut)
data.cumfreq = cumsum(data.freq)
freq_low <- cbind(data.cumfreq) 
dim(freq_low)
freq_l <- freq_low/nrow(data%>%filter(X0 < 0.33))
dim(freq_l)
#min(data$K.PNB[data$Class_O < 0.5])
breaks_high = seq(min(data$P.PNB[data$X0 > 0.33]), 
                  max(data$P.PNB[data$X0 > 0.33]),
                  by=abs(max(data$P.PNB[data$X0 > 0.33])/100))
data.cut = cut(data$P.PNB[data$X0 > 0.33], breaks_high, right=FALSE)
data.freq = table(data.cut)
data.cumfreq = cumsum(data.freq)
freq_high <- cbind(data.cumfreq)
dim(freq_high)
freq_h <- freq_high/nrow(data%>%filter(X0 > 0.33))
dim(freq_h)
ggplot()+
  geom_line(aes(x = c(1:123), y = freq_h), colour = "red")+
  geom_line(aes(x = c(1:123), y = freq_l), colour = "blue")

ks.test(freq_l, freq_h, alternative = "two.sided")

#----- N_PNB MCF--------------------------------------------
ggplot()+
  
  geom_density(aes(N.PNB),alpha = 0.3, fill = "blue", data = data%>%filter(X0 < 0.33))+
  #geom_histogram(aes(MutMxSz),alpha = 0.3, fill = "blue",bins = 50, data = data%>%filter(AvgMutPop > 70))+
  geom_density(aes(N.PNB),alpha = 0.3, fill = "red", data = data%>%filter(X0 > 0.33))+
  geom_vline(xintercept = -23)
min(data$N.PNB)
max(data$N.PNB)
breaks_low = seq(min(data$N.PNB[data$X0< 0.33]),max(data$N.PNB[data$X0 < 0.33]), 
                 by=abs(max((data$N.PNB[data$X0<= 0.33])/100)))
#max((data$K.PNB[data$Class_O > 0.5]))/100
data.cut = cut(data$N.PNB[data$X0 < 0.33], breaks_low, right=FALSE)
data.freq = table(data.cut)
data.cumfreq = cumsum(data.freq)
freq_low <- cbind(data.cumfreq) 
dim(freq_low)
freq_l <- freq_low/nrow(data%>%filter(X0 < 0.33))
dim(freq_l)
#min(data$K.PNB[data$Class_O < 0.5])
breaks_high = seq(min(data$N.PNB[data$X0 > 0.33]), 
                  max(data$N.PNB[data$X0 > 0.33]),
                  by=abs(max(data$N.PNB[data$X0 > 0.33])/100))
data.cut = cut(data$N.PNB[data$X0 > 0.33], breaks_high, right=FALSE)
data.freq = table(data.cut)
data.cumfreq = cumsum(data.freq)
freq_high <- cbind(data.cumfreq)
dim(freq_high)
freq_h <- freq_high/nrow(data%>%filter(X0 > 0.33))
dim(freq_h)
ggplot()+
  geom_line(aes(x = c(1:179), y = freq_h), colour = "red")+
  geom_line(aes(x = c(1:179), y = freq_l), colour = "blue")

ks.test(freq_l, freq_h, alternative = "two.sided")

# #-----Labor MCF--------------------------------------------
colnames(data)
ggplot()+
  geom_density(aes(Labor),alpha = 0.3, fill = "blue", data = data%>%filter(X0 < 0.33)) +
  #geom_histogram(aes(MutMxSz),alpha = 0.3, fill = "blue",bins = 50, data = data%>%filter(AvgMutPop > 70))+
  geom_density(aes(Labor),alpha = 0.3, fill = "red", data = data%>%filter(X0 > 0.33)) +
  geom_vline(xintercept = 28)
min(data$Labor)
max(data$Labor)
breaks_low = seq(min(data$Labor[data$X0 < 0.33]),max(data$Labor[data$X0 < 0.33]),
                 by=max((data$Labor[data$X0 < 0.33])/100))
#max((data$K.PNB[data$Class_O > 0.5]))/100
data.cut = cut(data$Labor[data$X0 <= 0.33], breaks_low, right=FALSE)
data.freq = table(data.cut)
data.cumfreq = cumsum(data.freq)
freq_low <- cbind(data.cumfreq)
dim(freq_low)
freq_l <- freq_low/nrow(data%>%filter(X0 < 0.33))
dim(freq_l)
breaks_high = seq(min(data$Labor[data$X0 > 0.33]),
                  max(data$Labor[data$X0> 0.33]),
                  by=max(data$Labor[data$X0 > 0.33])/100)
data.cut = cut(data$Labor[data$X0 > 0.33], breaks_high, right=FALSE)
data.freq = table(data.cut)
data.cumfreq = cumsum(data.freq)
freq_high <- cbind(data.cumfreq)
dim(freq_high)
freq_h <- freq_high/nrow(data%>%filter(X0 > 0.33))
dim(freq_h)

ggplot()+
  geom_line(aes(x = c(1:99), y = freq_h), colour = "red")+
  geom_line(aes(x = c(1:99), y = freq_l), colour = "blue")

ks.test(freq_l, freq_h, alternative = "two.sided")

# # #-----Gender ratio MCF--------------------------------------------
#*p<0.01
colnames(data)
ggplot()+
  geom_density(aes(Gender.ratio),alpha = 0.3, fill = "blue", data = data%>%filter(X0 < 0.33))+
  #geom_histogram(aes(MutMxSz),alpha = 0.3, fill = "blue",bins = 50, data = data%>%filter(AvgMutPop > 70))+
  geom_density(aes(Gender.ratio),alpha = 0.3, fill = "red", data = data%>%filter(X0 > 0.33)) +
  geom_vline(xintercept = 0.9)
min(data$Gender.ratio)
max(data$Gender.ratio)
breaks_low = seq(min(data$Gender.ratio[data$X0 < 0.33]),max(data$Gender.ratio[data$X0 < 0.33]),
                 by=max((data$Gender.ratio[data$X0< 0.33])/100))

data.cut = cut(data$Gender.ratio[data$X0 < 0.33], breaks_low, right=FALSE)
data.freq = table(data.cut)
data.cumfreq = cumsum(data.freq)
freq_low <- cbind(data.cumfreq)
dim(freq_low)
freq_l <- freq_low/nrow(data%>%filter(X0 < 0.33))
dim(freq_l)
breaks_high = seq(min(data$Gender.ratio[data$X0 > 0.33]),
                  max(data$Gender.ratio[data$X0 > 0.33]),
                  by=max(data$Gender.ratio[data$X0 > 0.33])/100)
data.cut = cut(data$Gender.ratio[data$X0 > 0.33], breaks_high, right=FALSE)
data.freq = table(data.cut)
data.cumfreq = cumsum(data.freq)
freq_high <- cbind(data.cumfreq)
dim(freq_high)
freq_h <- freq_high/nrow(data%>%filter(X0 > 0.33))
dim(freq_h)

ggplot()+
  geom_line(aes(x = c(1:98), y = freq_h), colour = "red")+
  geom_line(aes(x = c(1:98), y = freq_l), colour = "blue")

ks.test(freq_l, freq_h, alternative = "two.sided")

#-----HH dependency ratio MCF--------------------------------------------
colnames(data)
ggplot()+
  geom_density(aes(Household.dependency.ratio),alpha = 0.3, fill = "blue", data = data%>%filter(X0 < 0.25)) +
  #geom_histogram(aes(MutMxSz),alpha = 0.3, fill = "blue",bins = 50, data = data%>%filter(AvgMutPop > 70))+
  geom_density(aes(Household.dependency.ratio),alpha = 0.3, fill = "red", data = data%>%filter(X0 > 0.25)) +
  geom_vline(xintercept = 0.49)
min(data$Household.dependency.ratio)
max(data$Household.dependency.ratio)
breaks_low = seq(min(data$Household.dependency.ratio[data$X0 < 0.33]),max(data$Household.dependency.ratio[data$X0 < 0.5]), 
                 by=max((data$Household.dependency.ratio[data$X0< 0.33])/100))

data.cut = cut(data$Household.dependency.ratio[data$X0 < 0.33], breaks_low, right=FALSE)
data.freq = table(data.cut)
data.cumfreq = cumsum(data.freq)
freq_low <- cbind(data.cumfreq) 
dim(freq_low)
freq_l <- freq_low/nrow(data%>%filter(X0 < 0.33))
dim(freq_l)
breaks_high = seq(min(data$Household.dependency.ratio[data$X0 > 0.33]), 
                  max(data$Household.dependency.ratio[data$X0 > 0.33]),
                  by=max(data$Household.dependency.ratio[data$X0 > 0.33])/100)
data.cut = cut(data$Household.dependency.ratio[data$X0 > 0.33], breaks_high, right=FALSE)
data.freq = table(data.cut)
data.cumfreq = cumsum(data.freq)
freq_high <- cbind(data.cumfreq)
dim(freq_high)
freq_h <- freq_high/nrow(data%>%filter(X0 > 0.33))
dim(freq_h)

ggplot()+
  geom_line(aes(x = c(1:83), y = freq_h), colour = "red")+
  geom_line(aes(x = c(1:83), y = freq_l), colour = "blue")

ks.test(freq_l, freq_h, alternative = "two.sided")

# #-----Distance to roads MCF--------------------------------------------
colnames(data)
ggplot()+
  geom_density(aes(Distance.to.roads),alpha = 0.3, fill = "blue", data = data%>%filter(X0 < 0.33)) +
  #geom_histogram(aes(MutMxSz),alpha = 0.3, fill = "blue",bins = 50, data = data%>%filter(AvgMutPop > 70))+
  geom_density(aes(Distance.to.roads),alpha = 0.3, fill = "red", data = data%>%filter(X0 > 0.33)) +
  geom_vline(xintercept = 3.7)
min(data$Distance.to.roads)
max(data$Distance.to.roads)
breaks_low = seq(min(data$Distance.to.roads[data$X0 < 0.33]),max(data$Distance.to.roads[data$X0 < 0.33]),
                 by=max((data$Distance.to.roads[data$X0< 0.33])/100))

data.cut = cut(data$Distance.to.roads[data$X0 < 0.33], breaks_low, right=FALSE)
data.freq = table(data.cut)
data.cumfreq = cumsum(data.freq)
freq_low <- cbind(data.cumfreq)
dim(freq_low)
freq_l <- freq_low/nrow(data%>%filter(X0 < 0.33))
dim(freq_l)
breaks_high = seq(min(data$Distance.to.roads[data$X0 > 0.33]),
                  max(data$Distance.to.roads[data$X0 > 0.33]),
                  by=max(data$Distance.to.roads[data$X0 > 0.33])/100)
data.cut = cut(data$Distance.to.roads[data$X0 > 0.33], breaks_high, right=FALSE)
data.freq = table(data.cut)
data.cumfreq = cumsum(data.freq)
freq_high <- cbind(data.cumfreq)
dim(freq_high)
freq_h <- freq_high/nrow(data%>%filter(X0 > 0.33))
dim(freq_h)

ggplot()+
  geom_line(aes(x = c(1:100), y = freq_h), colour = "red")+
  geom_line(aes(x = c(1:100), y = freq_l), colour = "blue")

ks.test(freq_l, freq_h, alternative = "two.sided")

# #-----Distance to river MCF--------------------------------------------
colnames(data)
ggplot()+
  geom_density(aes(Distance.to.river),alpha = 0.3, fill = "blue", data = data%>%filter(X0 < 0.33)) +
  #geom_histogram(aes(MutMxSz),alpha = 0.3, fill = "blue",bins = 50, data = data%>%filter(AvgMutPop > 70))+
  geom_density(aes(Distance.to.river),alpha = 0.3, fill = "red", data = data%>%filter(X0 > 0.33)) +
  geom_vline(xintercept = 11.7)
min(data$Distance.to.river)
max(data$Distance.to.river)
breaks_low = seq(min(data$Distance.to.river[data$X0 < 0.33]),max(data$Distance.to.river[data$X0 < 0.33]),
                 by=max((data$Distance.to.river[data$X0< 0.33])/100))

data.cut = cut(data$Distance.to.river[data$X0 < 0.33], breaks_low, right=FALSE)
data.freq = table(data.cut)
data.cumfreq = cumsum(data.freq)
freq_low <- cbind(data.cumfreq)
dim(freq_low)
freq_l <- freq_low/nrow(data%>%filter(X0 < 0.33))
dim(freq_l)
breaks_high = seq(min(data$Distance.to.river[data$X0 > 0.33]),
                  max(data$Distance.to.river[data$X0 > 0.33]),
                  by=max(data$Distance.to.river[data$X0 > 0.33])/100)
data.cut = cut(data$Distance.to.river[data$X0 > 0.33], breaks_high, right=FALSE)
data.freq = table(data.cut)
data.cumfreq = cumsum(data.freq)
freq_high <- cbind(data.cumfreq)
dim(freq_high)
freq_h <- freq_high/nrow(data%>%filter(X0 > 0.33))
dim(freq_h)

ggplot()+
  geom_line(aes(x = c(1:98), y = freq_h), colour = "red")+
  geom_line(aes(x = c(1:98), y = freq_l), colour = "blue")

ks.test(freq_l, freq_h, alternative = "two.sided")
# #-----Distance to forest MCF--------------------------------------------
colnames(data)
ggplot()+
  geom_density(aes(Distance.to.forest),alpha = 0.3, fill = "blue", data = data%>%filter(X0 < 0.33)) +
  #geom_histogram(aes(MutMxSz),alpha = 0.3, fill = "blue",bins = 50, data = data%>%filter(AvgMutPop > 70))+
  geom_density(aes(Distance.to.forest),alpha = 0.3, fill = "red", data = data%>%filter(X0 > 0.33)) +
  geom_vline(xintercept = 20)
min(data$Distance.to.forest)
max(data$Distance.to.forest)
breaks_low = seq(min(data$Distance.to.forest[data$X0 < 0.33]),max(data$Distance.to.forest[data$X0 < 0.33]),
                 by=max((data$Distance.to.forest[data$X0< 0.33])/100))

data.cut = cut(data$Distance.to.forest[data$X0 < 0.33], breaks_low, right=FALSE)
data.freq = table(data.cut)
data.cumfreq = cumsum(data.freq)
freq_low <- cbind(data.cumfreq)
dim(freq_low)
freq_l <- freq_low/nrow(data%>%filter(X0 < 0.33))
dim(freq_l)
breaks_high = seq(min(data$Distance.to.forest[data$X0 > 0.33]),
                  max(data$Distance.to.forest[data$X0 > 0.33]),
                  by=max(data$Distance.to.forest[data$X0 > 0.33])/100)
data.cut = cut(data$Distance.to.forest[data$X0 > 0.33], breaks_high, right=FALSE)
data.freq = table(data.cut)
data.cumfreq = cumsum(data.freq)
freq_high <- cbind(data.cumfreq)
dim(freq_high)
freq_h <- freq_high/nrow(data%>%filter(X0 > 0.33))
dim(freq_h)

ggplot()+
  geom_line(aes(x = c(1:98), y = freq_h), colour = "red")+
  geom_line(aes(x = c(1:98), y = freq_l), colour = "blue")

ks.test(freq_l, freq_h, alternative = "two.sided")
#-----Carbon capacity MCF--------------------------------------------
# colnames(data)
# max(data$Carbon.capacity)
# ggplot()+
#   geom_density(aes(Carbon.capacity),alpha = 0.3, fill = "blue", data = data%>%filter(X0 < 0.33)) +
#   #geom_histogram(aes(MutMxSz),alpha = 0.3, fill = "blue",bins = 50, data = data%>%filter(AvgMutPop > 70))+
#   geom_density(aes(Carbon.capacity),alpha = 0.3, fill = "red", data = data%>%filter(X0 > 0.33)) +
#   geom_vline(xintercept = 47)
# min(data$Carbon.capacity)
# max(data$Carbon.capacity)
# breaks_low = seq(min(data$Carbon.capacity[data$X0 < 0.33]),max(data$Carbon.capacity[data$X0 < 0.33]),
#                  by=max((data$Carbon.capacity[data$X0< 0.33])/100))
# 
# data.cut = cut(data$Carbon.capacity[data$X0 < 0.33], breaks_low, right=FALSE)
# data.freq = table(data.cut)
# data.cumfreq = cumsum(data.freq)
# freq_low <- cbind(data.cumfreq)
# dim(freq_low)
# freq_l <- freq_low/nrow(data%>%filter(X0 < 0.33))
# dim(freq_l)
# breaks_high = seq(min(data$Carbon.capacity[data$X0 > 0.33]),
#                   max(data$Carbon.capacity[data$X0 > 0.33]),
#                   by=max(data$Carbon.capacity[data$X0 > 0.33])/100)
# data.cut = cut(data$Carbon.capacity[data$X0 > 0.33], breaks_high, right=FALSE)
# data.freq = table(data.cut)
# data.cumfreq = cumsum(data.freq)
# freq_high <- cbind(data.cumfreq)
# dim(freq_high)
# freq_h <- freq_high/nrow(data%>%filter(X0 > 0.33))
# dim(freq_h)
# 
# ggplot()+
#   geom_line(aes(x = c(1:82), y = freq_h), colour = "red")+
#   geom_line(aes(x = c(1:82), y = freq_l), colour = "blue")
# 
# ks.test(freq_l, freq_h, alternative = "two.sided")
#-----slope MCF--------------------------------------------
colnames(data)
max(data$Soil.slope)
ggplot()+
  geom_density(aes(Soil.slope),alpha = 0.3, fill = "blue", data = data%>%filter(X0 < 0.33)) +
  #geom_histogram(aes(MutMxSz),alpha = 0.3, fill = "blue",bins = 50, data = data%>%filter(AvgMutPop > 70))+
  geom_density(aes(Soil.slope),alpha = 0.3, fill = "red", data = data%>%filter(X0 > 0.33)) +
  geom_vline(xintercept = 2.4)
min(data$Soil.slope)
max(data$Soil.slope)
breaks_low = seq(min(data$Soil.slope[data$X0 < 0.33]),max(data$Soil.slope[data$X0 < 0.33]),
                 by=max((data$Soil.slope[data$X0< 0.33])/100))

data.cut = cut(data$Soil.slope[data$X0 < 0.33], breaks_low, right=FALSE)
data.freq = table(data.cut)
data.cumfreq = cumsum(data.freq)
freq_low <- cbind(data.cumfreq)
dim(freq_low)
freq_l <- freq_low/nrow(data%>%filter(X0 < 0.33))
dim(freq_l)
breaks_high = seq(min(data$Soil.slope[data$X0 > 0.33]),
                  max(data$Soil.slope[data$X0 > 0.33]),
                  by=max(data$Soil.slope[data$X0 > 0.33])/100)
data.cut = cut(data$Soil.slope[data$X0 > 0.33], breaks_high, right=FALSE)
data.freq = table(data.cut)
data.cumfreq = cumsum(data.freq)
freq_high <- cbind(data.cumfreq)
dim(freq_high)
freq_h <- freq_high/nrow(data%>%filter(X0 > 0.33))
dim(freq_h)

ggplot()+
  geom_line(aes(x = c(1:99), y = freq_h), colour = "red")+
  geom_line(aes(x = c(1:99), y = freq_l), colour = "blue")

ks.test(freq_l, freq_h, alternative = "two.sided")

#-----Trees MCF--------------------------------------------
colnames(data)
max(data$Tree.richness)
ggplot()+
  geom_density(aes(Tree.richness),alpha = 0.3, fill = "blue", data = data%>%filter(X0 < 0.33)) +
  #geom_histogram(aes(MutMxSz),alpha = 0.3, fill = "blue",bins = 50, data = data%>%filter(AvgMutPop > 70))+
  geom_density(aes(Tree.richness),alpha = 0.3, fill = "red", data = data%>%filter(X0 > 0.33)) +
  geom_vline(xintercept = 7.5)
min(data$Tree.richness)
max(data$Tree.richness)
breaks_low = seq(min(data$Tree.richness[data$X0 < 0.33]),max(data$Tree.richness[data$X0 < 0.33]),
                 by=max((data$Tree.richness[data$X0< 0.33])/100))

data.cut = cut(data$Tree.richness[data$X0 < 0.33], breaks_low, right=FALSE)
data.freq = table(data.cut)
data.cumfreq = cumsum(data.freq)
freq_low <- cbind(data.cumfreq)
dim(freq_low)
freq_l <- freq_low/nrow(data%>%filter(X0 < 0.33))
dim(freq_l)
breaks_high = seq(min(data$Tree.richness[data$X0 > 0.33]),
                  max(data$Tree.richness[data$X0 > 0.33]),
                  by=max(data$Tree.richness[data$X0 > 0.33])/100)
data.cut = cut(data$Tree.richness[data$X0 > 0.33], breaks_high, right=FALSE)
data.freq = table(data.cut)
data.cumfreq = cumsum(data.freq)
freq_high <- cbind(data.cumfreq)
dim(freq_high)
freq_h <- freq_high/nrow(data%>%filter(X0 > 0.33))
dim(freq_h)

ggplot()+
  geom_line(aes(x = c(1:99), y = freq_h), colour = "red")+
  geom_line(aes(x = c(1:99), y = freq_l), colour = "blue")

ks.test(freq_l, freq_h, alternative = "two.sided")

#-----Input Intensity--------------------------------------------
colnames(data)
max(data$Input.intensity)
ggplot()+
  geom_density(aes(Input.intensity),alpha = 0.3, fill = "blue", data = data%>%filter(X0 < 0.33)) +
  #geom_histogram(aes(MutMxSz),alpha = 0.3, fill = "blue",bins = 50, data = data%>%filter(AvgMutPop > 70))+
  geom_density(aes(Input.intensity),alpha = 0.3, fill = "red", data = data%>%filter(X0 > 0.33)) +
  geom_vline(xintercept = 1.3)
min(data$Input.intensity)
max(data$Input.intensity)
breaks_low = seq(min(data$Input.intensity[data$X0 < 0.33]),max(data$Input.intensity[data$X0 < 0.33]),
                 by=max((data$Input.intensity[data$X0< 0.33])/100))

data.cut = cut(data$Input.intensity[data$X0 < 0.33], breaks_low, right=FALSE)
data.freq = table(data.cut)
data.cumfreq = cumsum(data.freq)
freq_low <- cbind(data.cumfreq)
dim(freq_low)
freq_l <- freq_low/nrow(data%>%filter(X0 < 0.33))
dim(freq_l)
breaks_high = seq(min(data$Input.intensity[data$X0 > 0.33]),
                  max(data$Input.intensity[data$X0 > 0.33]),
                  by=max(data$Input.intensity[data$X0 > 0.33])/100)
data.cut = cut(data$Input.intensity[data$X0 > 0.33], breaks_high, right=FALSE)
data.freq = table(data.cut)
data.cumfreq = cumsum(data.freq)
freq_high <- cbind(data.cumfreq)
dim(freq_high)
freq_h <- freq_high/nrow(data%>%filter(X0 > 0.33))
dim(freq_h)

ggplot()+
  geom_line(aes(x = c(1:92), y = freq_h), colour = "red")+
  geom_line(aes(x = c(1:92), y = freq_l), colour = "blue")

ks.test(freq_l, freq_h, alternative = "two.sided")



