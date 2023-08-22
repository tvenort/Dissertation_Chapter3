#Read data file
data<-read.csv("./8_Analyses/ML_GSA/ml_maize_data_final.csv")
colnames(data)
#yield_bp
library(viridis)
library(RColorBrewer)
library(wesanderson)
#--yield variability plots by class-------
data$class<-data$Maize_yield_class
data$Yield<-data$Maize_yield/1000
data<-as.data.frame(data)
data
data$Class<-ifelse(data$class == 0,'Low-productivity',
                               ifelse(data$class ==1,'Mid-productivity',
                                      ifelse(data$class ==2,'High-productivity',"")))
data
p <- ggplot(data,aes(x=factor(Class, level=c("Low-productivity","Mid-productivity",
                                                         "High-productivity")),y=Yield,))+
geom_boxplot() +geom_hline(yintercept=1.5,linetype="dashed", color = "grey")+
geom_hline(yintercept=2.5,linetype="dashed", color = "grey")+
  xlab('Class') + ylab('Maize yield (t/ha)')
p<-p+ geom_jitter(shape=16, position=position_jitter(0.2))
p
coord<-read.csv(paste('./8_Analyses/Spatial_analyses/Eplots_coordinates.csv', sep=""))
data
#distribution of classes accross the landscapes
#Landscape 3
cv_l3<-filter(data, landscape_no == 'L03')
cv_l3
cv_l3<-cv_l3%>%count(Class)
cv_l3
#Landscape 10
cv_l10<-filter(data, landscape_no == 'L10')
cv_l10
cv_l10<-cv_l10%>%count(Class)
cv_l10
#Landscape 11
cv_l11<-filter(data, landscape_no == 'L11')
cv_l11
cv_l11<-cv_l11%>%count(Class)
cv_l11
# #Landscape 18
cv_l18<-filter(data, landscape_no == 'L18')
cv_l18
cv_l18<-cv_l18%>%count(Class)
cv_l18
#Landscape 19
cv_l19<-filter(data, landscape_no == 'L19')
cv_l19
cv_l19<-cv_l19%>%count(Class)
cv_l19
#Landscape 20
cv_l20<-filter(data, landscape_no == 'L20')
cv_l20
cv_l20<-cv_l20%>%count(Class)
cv_l20
#Landscape 22
cv_l22<-filter(data, landscape_no == 'L22')
cv_l22
cv_l22<-cv_l22%>%count(Class)
cv_l22

#---K_PNB by class------
p1 <- ggplot(data,aes(x=factor(Class, level=c("Low-productivity","Mid-productivity",
                                                 "High-productivity")),y=K_PNB,))+
  geom_boxplot() +
  xlab('Class') + ylab('K.PNB')+ 
  geom_jitter(shape=16, position=position_jitter(0.2))
p1

ggsave("k.pnb_boxplot.jpg", p1,
      # width =20, height=32, units =c("cm"),
       dpi=300,path = "./8_Analyses/Plots/Features_class_boxplots")

#-----P_PNB_class
ppnb<-c()
ppnb$class<-data$Maize_yield_class
ppnb$p_pnb<-data$P_PNB
ppnb<-as.data.frame(ppnb)
ppnb
ppnb$Class<-ifelse(ppnb$class == 0,'Low-productivity',
                   ifelse(ppnb$class ==1,'Mid-productivity',
                          ifelse(ppnb$class ==2,'High-productivity',"")))
ppnb
p2<- ggplot(ppnb,aes(x=factor(Class, level=c("Low-productivity","Mid-productivity",
                                              "High-productivity")),y=p_pnb,))+
  geom_boxplot() +
  xlab('Class') + ylab('K.PNB')+ 
  geom_jitter(shape=16, position=position_jitter(0.2))
p2

ggsave("p.pnb_boxplot.jpg", p2,
       # width =20, height=32, units =c("cm"),
       dpi=300,path = "./8_Analyses/Plots/Features_class_boxplots")

#-----N_PNB_class
npnb<-c()
npnb$class<-data$Maize_yield_class
npnb$n_pnb<-data$N_PNB
npnb<-as.data.frame(npnb)
npnb
npnb$Class<-ifelse(npnb$class == 0,'Low-productivity',
                   ifelse(npnb$class ==1,'Mid-productivity',
                          ifelse(npnb$class ==2,'High-productivity',"")))
npnb
p3<- ggplot(npnb,aes(x=factor(Class, level=c("Low-productivity","Mid-productivity",
                                             "High-productivity")),y=n_pnb,))+
  geom_boxplot() +
  xlab('Class') + ylab('N.PNB')+ 
  geom_jitter(shape=16, position=position_jitter(0.2))
p3

ggsave("n.pnb_boxplot.jpg", p3,
       # width =20, height=32, units =c("cm"),
       dpi=300,path = "./8_Analyses/Plots/Features_class_boxplots")


#-----Soil_slope----
ss<-c()
ss$class<-data$Maize_yield_class
ss$ss<-data$Slope
ss<-as.data.frame(ss)
ss
ss$Class<-ifelse(ss$class == 0,'Low-productivity',
                   ifelse(ss$class ==1,'Mid-productivity',
                          ifelse(ss$class ==2,'High-productivity',"")))
ss
p4<- ggplot(ss,aes(x=factor(Class, level=c("Low-productivity","Mid-productivity",
                                             "High-productivity")),y=ss,))+
  geom_boxplot() +
  xlab('Class') + ylab('Soil.slope')+ 
  geom_jitter(shape=16, position=position_jitter(0.2))
p4

ggsave("slope_boxplot.jpg", p4,
       # width =20, height=32, units =c("cm"),
       dpi=300,path = "./8_Analyses/Plots/Features_class_boxplots")


