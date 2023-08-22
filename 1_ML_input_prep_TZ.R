#######################################################
# SILL HH  landscape/livelihoods/Mgmt indicators   ###
######################################################
#--Setups-------
#standardizing functions
lib<-function(x){
  y<-1-((x-min(x))/(max(x)-min(x)))
  return(y)
}
mib<-function(x){
  y<-(x-min(x))/(max(x)-min(x))
  return(y)
}
opt_pH<-function(x){
  y<-1-((abs(x-7) - abs(min(x)-7))/(abs(max(x)-7) -abs(min(x)-7)))
  return(y)
}
opt_bd<-function(x){
  y<-abs(1-(abs(x-130) - abs(min(x)-130))/(abs(max(x)-130) -abs(min(x)-130)))
  return(y)
}
#----Maize data------
#loading maize es data
maize_yield<-read.csv('./8_Analyses/ML_GSA/datasets/ml_es_maize_knnimp.csv')
colnames(maize_yield)
dim(maize_yield)
maize_yield<-maize_yield[c(1,5)]
library(tidyverse)
maize_yield<-filter_if(maize_yield, is.numeric, all_vars((.) != 0))
maize_yield
#soil quality PCs
soils<-read.csv('./8_Analyses/ML_GSA/datasets/ml_es_maize_knnimp.csv')
colnames(soils)
soils<-soils[,-c(5)]
soils
# #loading lcdm assets data
lcdm<- read.csv('./8_Analyses/Datasets/HH_masterdata_knnimp.csv')
colnames(lcdm)
dim(lcdm)
head(lcdm)
colnames(lcdm)
View(lcdm)
#View(lcdm)
lcdm
#oldvals <- c(0,1)
#newvals <- c(1,0)
#lcdm$res_ret<-newvals[match(lcdm$no_resret, oldvals)]
colnames(lcdm)
lcdm<-lcdm[c("hh_refno","ave_precip","dist_water_bodies","Shannon_tree","richness_tree","Slope","Elevation",
           "dist_forest","Male_Female_ratio","hhmembers","hh_dep_ratio","literacy_rate",
            "leveledu","hh_labor_days","Info_network","market_network","ag_implements",
           "water_inf","farm_structures","comm_inf","dist_cell_wifi_tower","transp_inf",
            "dist_main_roads","dist_any_roads","time_field_travel","wage_entry_n","liv_sale_number",
            "house_tenure_n","house_walls_n","house_roof_n","house_floor_n","house_san_n",
           "house_electricity_n","house_water_n","house_fuel_cook_n","house_fuel_light_n","hh_microf_n",
            "landsize","farm_type","fields", "CIF")]

#merging the three
library(dplyr)
data<-lcdm%>%left_join(maize_yield,by='hh_refno')%>%left_join(soils,
        by='hh_refno')
data<-na.omit(data)
dim(data)
colnames(data)
View(data)
#View(data)
#mising data
as.data.frame(colSums(is.na(data[,-c(1)])))
#terciles
library(tidyverse)
# maize yield curation
#remove yield > 2200 kg/ha based on max smallholder yield range in SSA
library(dplyr)
#data<-filter(data,data$Maize_yield < 2500)
dim(data)
colnames(data)
range(data$dist_water_bodies)
range(data$dist_forest)
range(data$Slope.x)
range(data$richness_tree)
range(data$Male_Female_ratio)
range(data$hh_dep_ratio)
range(data$hh_labor_days)
range(data$dist_any_roads)
range(data$CIF)
range(data$N_PNB)
range(data$P_PNB)
range(data$K_PNB)
median(data$N_PNB)
median(data$P_PNB)
median(data$K_PNB)
range(data$C_capacity)
range(data$wage_entry_n)
#data <-data[-c(185),]
#View(data)
#1446 is average yield in SSA, 2500 is the attainable yield in assisted countries
#OR Kenya specific 1000 kg/ha katterer et al
#data$maize_yield_percent<-data$Maize_yield/2500
#quantile(data$maize_yield)
#quantile(data$maize_yield_percent)
range(data$Maize_yield)
mean(data$Maize_yield)
#divide maize percent into 3 quantiles based on maize yield or maize yield percentage
#data<-data%>%mutate(maize_yield_percent_Q = ntile(maize_yield_percent, 3))
#data$maize_yield_percent_Q
#classification
data$Maize_yield_class<-ifelse(data$Maize_yield <= 1500,0,
                        ifelse(data$Maize_yield > 1500 & data$Maize_yield <=2500,1,
                        ifelse(data$Maize_yield > 2500,2,"")))
# data$maize_yield_percent_class<-ifelse(data$maize_yield_percent < 0.7,0,
#               ifelse(data$maize_yield_percent >= 0.7 & data$maize_yield_percent <= 1,1,
#               ifelse(data$maize_yield_percent >1,2,"")))
# as.data.frame(colSums(is.na(data[,-c(1)])))
table(data$Maize_yield_class)
colnames(data)
dim(data)
min(data$Maize_yield)
write.csv(data,'./8_Analyses/ML_GSA/ml_maize_data_final.csv',row.names = FALSE)
colnames(data)
final_data<-data[c("hh_refno","ave_precip","dist_water_bodies","Shannon_tree","richness_tree","Slope",
                   "dist_forest","Male_Female_ratio","hh_dep_ratio",
                   "hh_labor_days","dist_cell_wifi_tower",
                   "dist_main_roads","CIF","N_PNB","P_PNB","K_PNB","Maize_yield")]
colnames(final_data)
names(final_data)[2]<-'Precipitation'
names(final_data)[3]<-'Distance.to.river'
names(final_data)[4]<-'Tree.diversity.shannon'
names(final_data)[5]<-'Tree.richness'
names(final_data)[6]<-'Soil.slope'
names(final_data)[7]<-'Distance.to.forest'
names(final_data)[8]<-'Gender.ratio'
names(final_data)[9]<-'Household.dependency.ratio'
names(final_data)[10]<-'labor'
names(final_data)[11]<-'Distance.to.wifi.tower'
names(final_data)[12]<-'Distance.to.roads'
names(final_data)[13]<-'Input.Intensity.index'
names(final_data)[14]<-'N.PNB'
names(final_data)[15]<-'P.PNB'
names(final_data)[16]<-'K.PNB'
names(final_data)[17]<-'Maize.yield'
final_data
write.csv(final_data,'./8_Analyses/ML_GSA/final_data_app.csv',row.names = FALSE)
data
#yield_bp
library(viridis)
library(RColorBrewer)
library(wesanderson)
#--yield by class-------
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


#--Boxplots for each retained feature of the model-----
#---Rice data------------
#loading rice es data
# rice_yield<- read.csv('./8_Analyses/ML_models/datasets/ml_es_rice_knnimp.csv')
# colnames(rice_yield)
# dim(rice_yield)
# #soil quality PCs
# soils<-read.csv('./8_Analyses/ML_models/datasets/ml_es_rice_knnimp.csv')
# soils<-soils[c(1,3:23)]
# colnames(soils)
# # sq<-soils[c("hh_refno","sulfur","pH","Clay","Ca","Mg",      
# #             "C_capacity","N_PNB","P_PNB","K_PNB")]
# # sq
# # sq$P_PNB<-mib(soils$P_PNB)
# # sq$sulfur<-lib(soils$sulfur)
# # sq$pH<-opt_pH(soils$pH)
# # sq$Clay<-mib(soils$Clay)
# # sq$Mg<-mib(soils$Mg)
# # sq$K_PNB<-mib(soils$K_PNB)
# # sq$C_capacity<-mib(soils$C_capacity)
# # sq$N_PNB<-mib(soils$N_PNB)
# # library(FactoMineR)
# # library(factoextra)
# # sq_scaled<- scale(sq[,-c(1)])
# # corr_matrix <-cor(sq_scaled)
# # library(ggcorrplot)
# # #ggcorrplot(corr_matrix)
# # colnames(sq_scaled)
# # sq.pca <- PCA(sq_scaled, graph = FALSE)
# # sq.pca$var$contrib
# # #identifying PCs with >70 % loading
# # eigenvalues <-sq.pca$eig
# # head(eigenvalues[, 1:3])
# # sq_pca<-sq.pca$ind$coord
# # sq_pca<-sq_pca[,c(1:3)]
# # sq_pca<-cbind(sq$hh_refno,sq_pca)
# # sq_pca<-as.data.frame(sq_pca)
# # names(sq_pca)[1]<-'hh_refno'
# # sq_pca
# #merging of datasets
# library(dplyr)
# data<-lci%>%left_join(rice_yield,by='hh_refno')%>%left_join(soils,
#       by='hh_refno')%>%left_join(lcdm,by='hh_refno')
# data<-na.omit(data)
# dim(data)
# colnames(data)
# #mising data
# as.data.frame(colSums(is.na(data[,-c(1)])))
# library(tidyverse)
# colnames(data)
# #1.38t/ha is the attainable yield in SSA -Sanchez; 2100 average for SSA
# #find regional average
# data$Rice_yield_percent<-data$Rice_yield/1380
# data$Rice_yield_percent
# range(data$Rice_yield_percent)
# #divide maize percent into 3 quantiles based on maize yield or maize yield percentage
# #data<-data%>%mutate(Rice_yield_percent_Q = ntile(Rice_yield_percent, 3))
# #classification
# data$Rice_yield_percent_class<-ifelse(data$Rice_yield_percent < 0.7,0,
#                     ifelse(data$Rice_yield_percent >= 0.7 & data$Rice_yield_percent <=1,1,
#                     ifelse(data$Rice_yield_percent >1,2,"")))
# table(data$Rice_yield_percent_class)
# #as.data.frame(colSums(is.na(data[,-c(1)])))
# View(data)
# write.csv(data,'./8_Analyses/ML_models/ml_rice_data_final.csv',row.names = FALSE)
