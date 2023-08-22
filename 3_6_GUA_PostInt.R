library(tidyverse)
library(ggplot2)
##---Pre-intervention class probabilities---------------------------------------
#--For sobol sample -----
#sobol sample
class_sobol<-read.csv(paste('./8_Analyses/ML_GSA/gsa_runs.csv',sep=""))
colnames(class_sobol)
dim(class_sobol)
t<-569500
class_sobol<-class_sobol%>%count(X0)
class_sobol$percent<-(class_sobol$n/t)*100
class_sobol<-as.data.frame(class_sobol)
names(class_sobol)[1]<-'Class'
class_sobol$Class[1]<-'Low productivity'
class_sobol$Class[2]<-'Mid productivity'
class_sobol$Class[3]<-'High productivity'
class_sobol
# p <- ggplot(class_sobol, aes(x=Class,y=percent,fill=Class)) +
#   geom_bar(stat='identity')
# p
# p+ylab('Percent')+ xlab('Class') +
# geom_text(aes(label =percent ), size = 3, hjust = 0.5,
#           vjust = 1, position = "stack")

#Read model outputs
class1<- read.csv(paste('./8_Analyses/ML_GSA/gsa_runs0.csv', sep=""))
names(class1)[1]<-'N'
names(class1)[2]<-'Probability_class1'
colnames(class1)
dim(class1)
range(class1$Probability)
# library(ggplot2)
# p <- ggplot(class1, aes(x=N,y=Probability_lp)) + 
#   geom_boxplot(varwidth = TRUE)
# p + geom_hline(yintercept=0.25,linetype="dashed", color = "black")+
#   ylab('Probability of low productivity') + xlab('Sample (N=753664)')
# median(class1$Probability_class1)
class2<- read.csv(paste('./8_Analyses/ML_GSA/gsa_runs1.csv', sep=""))
names(class2)[2]<-'Probability_class2'
colnames(class2)
range(class2$Probability)
class3<- read.csv(paste('./8_Analyses/ML_GSA/gsa_runs2.csv', sep=""))
names(class3)[2]<-'Probability_class3'
colnames(class3)
range(class3$Probability)
class_gua<-cbind(class1,class2,class3)
class_gua<-class_gua[c(1,2,4,6)]
class_gua
#class_gua<-rbind(class1,class2,class3)
colnames(class_gua)
#Filtering probabilities
class1<-class_gua%>%filter(Probability_class1 > Probability_class2 & Probability_class1 >
                             Probability_class3)
count(class1)
class1$Class<-'Low-productivity (sobol)'
class1<-class1[c("Class","Probability_class1")]
dim(class1)
names(class1)[2]<-'Probability'
median(class1$Probability)
class2<-class_gua%>%filter(Probability_class2 > Probability_class1 & Probability_class2 >
                             Probability_class3)
class2$Class<-'Mid-productivity (sobol)'
dim(class2)
class2<-class2[c("Class","Probability_class2")]
names(class2)[2]<-'Probability'
median(class2$Probability)
class3<-class_gua%>%filter(Probability_class3 > Probability_class1 & Probability_class3 >
                             Probability_class2)
class3$Class<-'High-productivity (sobol)'
dim(class3)
View(class3)
class3<-class3[c("Class","Probability_class2")]
names(class3)[2]<-'Probability'
median(class3$Probability)
class_gua_s<-rbind(class1,class2,class3)
colnames(class_gua_s)
# library(ggplot2)
# p <- ggplot(class_gua, aes(x=factor(Class, level=c("Low productivity",
#                 "Mid productivity",
#                 "High productivity")),y=Probability)) + 
#       geom_boxplot(varwidth = TRUE)+
#   scale_x_discrete(labels=c("Low productivity (N=457194)",
#                             "Mid productivity (N=82239) ",
#                             "High productivity (N=30067)"))+
#   xlab('Sample (N =569500)')+
#  geom_hline(yintercept=0.33,linetype="dashed", color = "black")
# p
##-- For original sample-----
#raw data predictions 
raw_data<-read.csv(paste('./8_Analyses/ML_GSA/os_prob_outc_pre_int.csv', sep=""))
raw_data
raw_data<-raw_data[c("Class")]
raw_data<-raw_data%>%count(Class)
raw_data$percent<-(raw_data$n/152)*100
raw_data
raw_data_prob<-read.csv(paste('./8_Analyses/ML_GSA/os_prob_pre_int.csv', sep=""))
raw_data_prob
range(raw_data_prob$Class_O)
range(raw_data_prob$Class_1)
range(raw_data_prob$Class_2)
names(raw_data_prob)[1]<-'Probability_class1'
names(raw_data_prob)[2]<-'Probability_class2'
names(raw_data_prob)[3]<-'Probability_class3'
raw_data_prob
#Filtering probabilities
class1<-raw_data_prob%>%filter(Probability_class1 > Probability_class2 & Probability_class1 >
                             Probability_class3)
count(class1)
class1$Class<-'Low-productivity (original)'
class1<-class1[c("Class","Probability_class1")]
dim(class1)
median(class1$Probability_class1)
names(class1)[2]<-'Probability'
class2<-raw_data_prob%>%filter(Probability_class2 > Probability_class1 & Probability_class2 >
                             Probability_class3)
class2$Class<-'Mid-productivity (original)'
dim(class2)
median(class2$Probability_class2)
class2<-class2[c("Class","Probability_class2")]
names(class2)[2]<-'Probability'
class3<-raw_data_prob%>%filter(Probability_class3 > Probability_class1 & Probability_class3 >
                             Probability_class2)
class3$Class<-'High-productivity (original)'
dim(class3)
median(class3$Probability_class3)
class3<-class3[c("Class","Probability_class2")]
names(class3)[2]<-'Probability'
class_gua_o<-rbind(class1,class2,class3)
colnames(class_gua_o)
View(class_gua_o)
#binding sobol and original datasets
class_gua_all<-rbind(class_gua_o,class_gua_s)
colnames(class_gua_all)
class_gua_all
library(ggplot2)
p <- ggplot(class_gua_all, aes(x=factor(Class, level=c("Low-productivity (original)",
                "Low-productivity (sobol)",                                      
                "Mid-productivity (original)",
                "Mid-productivity (sobol)",
                "High-productivity (original)",
                "High-productivity (sobol)"))
                ,y=Probability)) +
      geom_boxplot()+
  scale_x_discrete(labels=c("LP (Original)",
                            "LP (Sobol)",
                            "MP (Original) ",
                            "MP (Sobol) ",
                            "HP (Original)",
                            "HP (Sobol)"))+
  xlab('Samples')+ylab('Probability')+
  stat_summary(fun.y = "median", geom = "point", shape = 23, size = 3, fill = "white")+
 geom_hline(yintercept=0.33,linetype="dashed", color = "black")
p
wilcox.test(class_gua_o$Probability,class_gua_s$Probability)
#-Pre-intervention class ------------------------------------------------------
#Read model outputs
pre<- read.csv(paste('./8_Analyses/ML_GSA/gsa_runs0.csv', sep=""))
pre$Scenario<-'Pre-INT'
pre<-pre[c('Scenario',"X0")]
names(pre)[2]<-'Probability'
colnames(pre)
median(pre$Probability)
#Pre-IS
#Read model outputs
pre_is<- read.csv(paste('./8_Analyses/ML_GSA/gsa_runs0.csv', sep=""))
pre_is$Scenario<-'Pre-IS'
pre_is<-pre_is[c('Scenario',"X0")]
names(pre_is)[2]<-'Probability'
colnames(pre_is)
median(pre_is$Probability)
#---Post K_PNB intervention-----------------------------------------------------
K<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/prob_k_pnb.csv', sep=""))
K$Scenario<-'Post-INT[K_pnb]'
K<-K[c('Scenario',"Class_O")]
names(K)[2]<-'Probability'
colnames(K)
dim(K)
median(K$Probability)

#---Post P_PNB intervention-----------------------------------------------------
P<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/prob_p_pnb.csv', sep=""))
P$Scenario<-'Post-INT[P_pnb]'
P<-P[c('Scenario',"Class_O")]
names(P)[2]<-'Probability'
colnames(P)
median(P$Probability)

#---Post N_PNB intervention-----------------------------------------------------
N<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/prob_n_pnb.csv', sep=""))
N$Scenario<-'Post-INT[N_pnb]'
N<-N[c('Scenario',"Class_O")]
names(N)[2]<-'Probability'
colnames(N)
mean(N$Probability)

#---Post fertilizer interventions-----------------------------------------------
fert<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/prob_fert.csv', sep=""))
fert$Scenario<-'Post-INT[combined NM factors]'
fert<-fert[c('Scenario',"Class_O")]
names(fert)[2]<-'Probability'
colnames(fert)
mean(fert$Probability)

#---Labor intervention-----------------------------------------------------
L<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/prob_labor.csv', sep=""))
L$Scenario<-'Post-INT[Labor]'
L<-L[c('Scenario',"Class_O")]
names(L)[2]<-'Probability'
colnames(L)
mean(L$Probability)
#---Dependency intervention-----------------------------------------------------
D<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/prob_hh_dep.csv', sep=""))
D$Scenario<-'Post-INT[HH_dep]'
D<-D[c('Scenario',"Class_O")]
names(D)[2]<-'Probability'
colnames(D)
mean(D$Probability)

#---Input Intensity-------------------------------------------------------------
ii<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/prob_input_int.csv', sep=""))
ii$Scenario<-'Post-INT[Input_intensity]'
ii<-ii[c('Scenario',"Class_O")]
names(ii)[2]<-'Probability'
colnames(ii)
mean(ii$Probability)

#---Household intervention----------------------------------------------------------
hh<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/prob_hhc.csv', sep=""))
hh$Scenario<-'Post-INT[combined HH factors]'
hh<-hh[c('Scenario',"Class_O")]
names(hh)[2]<-'Probability'
colnames(hh)
mean(hh$Probability)
#---River intervention----------------------------------------------------------
R<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/prob_dist_riv.csv', sep=""))
R$Scenario<-'Post-INT[dist_river]'
R<-R[c('Scenario',"Class_O")]
names(R)[2]<-'Probability'
colnames(R)
mean(R$Probability)
#---Roads intervention----------------------------------------------------------
Rd<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/prob_dist_roads.csv', sep=""))
Rd$Scenario<-'Post-INT[dist_roads]'
Rd<-Rd[c('Scenario',"Class_O")]
names(Rd)[2]<-'Probability'
colnames(Rd)
mean(Rd$Probability)
#---Slope intervention----------------------------------------------------------
S<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/prob_slope.csv', sep=""))
S$Scenario<-'Post-INT[soil_percent_slope]'
S<-S[c('Scenario',"Class_O")]
names(S)[2]<-'Probability'
colnames(S)
mean(S$Probability)
#---Landscape infra intervention----------------------------------------------------------
linf<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/prob_linf.csv', sep=""))
linf$Scenario<-'Post-INT[combined LANDS factors]'
linf<-linf[c('Scenario',"Class_O")]
names(linf)[2]<-'Probability'
colnames(linf)
mean(linf$Probability)

#---IS1------------(nutrient imbalance remediation)
is1<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/prob_fert.csv', sep=""))
is1$Scenario<-'IS1 [Nutrient imbalance remediation]'
is1<-is1[c('Scenario',"Class_O")]
names(is1)[2]<-'Probability'
colnames(is1)
mean(is1$Probability)

#---IS2------------(Soil health)
is2<-read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/prob_sh.csv', sep=""))
is2$Scenario<-'IS2 [Nutrient imbalance remediation + improved soil health]'
is2<-is2[c('Scenario',"Class_O")]
names(is2)[2]<-'Probability'
colnames(is2)
mean(is2$Probability)

#--IS3----------------(Soil health+ hh)
is3<-read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/prob_sh_hh.csv', sep=""))
is3$Scenario<-'IS3 [Nutrient imbalance remediation + improved soil health + increased household human and physical capital]'
is3<-is3[c('Scenario',"Class_O")]
names(is3)[2]<-'Probability'
colnames(is3)
mean(is3$Probability)

#--IS4-----------------------(Soil health + hh + lands)
is4<-read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/prob_sh_hh_ls.csv', sep=""))
is4$Scenario<-'IS4 [Nutrient imbalance remediation + improved soil health + increased household human and physical capital + increased landscape physical capital]'
is4<-is4[c('Scenario',"Class_O")]
names(is4)[2]<-'Probability'
colnames(is4)
mean(is4$Probability)
is4
#--Plots---------------------------------------------------------------
library(viridis)
library(RColorBrewer)
library(wesanderson)
library(ggplot2)
#install.packages("stringr")          # Install stringr package
library(stringr)  
#---nutrient management interventions-----
nmi<-rbind(pre,K,P,N,fert)
#interventions$Scenario<- factor(interventions$Scenario, levels = interventions$Scenario)
colnames(nmi)
# Basic box plot
nmi_p<-ggplot(nmi,aes(x=factor(Scenario,level=c("Pre-INT",
                               "Post-INT[K_pnb]",
                               "Post-INT[P_pnb]",
                               "Post-INT[N_pnb]",
       "Post-INT[combined NM factors]")),Scenario, y=Probability))+
  geom_boxplot(notch = TRUE) 
nmi_p + geom_hline(yintercept=0.33,linetype="dashed", color = "black")+ 
  xlab('Nutrient management interventions')+
  ylab('Probabilty of low productivity')
#ggsave("nmi_p.pdf")
#-----landscape infrastructure interventions----------
infi<-rbind(pre,R,Rd,S)
colnames(infi)
# Basic box plot
infi_p<- ggplot(infi,aes(x=factor(Scenario,level=c("Pre-INT",
                                                  "Post-INT[dist_river]",
                                                  "Post-INT[dist_roads]",
                                                 "Post-INT[soil_percent_slope]"
                                               )),Scenario,y=Probability
                                                       ))+
  geom_boxplot(notch = TRUE)
infi_p + geom_hline(yintercept=0.33,linetype="dashed", color = "black")+
  xlab('Landscape interventions')+
  ylab('Probabilty of low productivity')

#-----household interventions-------------------
hhi<-rbind(pre,L,D)
colnames(infi)
# Basic box plot
hhi_p<- ggplot(hhi,aes(x=factor(Scenario,level=c("Pre-INT",
                                                  "Post-INT[Labor]",
                                          "Post-INT[HH_dep]"
                                           )),
                                  Scenario,y=Probability))+
geom_boxplot()
hhi_p + geom_hline(yintercept=0.33,linetype="dashed", color = "black")+ 
  xlab('Household interventions')+ 
  ylab('Probabilty of low productivity')


#--All interventions-------
library(scales)
alli<-rbind(pre_is,is1,is2,is3,is4)
colnames(alli)
# Basic box plot
alli_p<- ggplot(alli,aes(x=factor(Scenario,level=c("Pre-IS",
            'IS1 [Nutrient imbalance remediation]',
            'IS2 [Nutrient imbalance remediation + improved soil health]',
            'IS3 [Nutrient imbalance remediation + improved soil health + increased household human and physical capital]',
            'IS4 [Nutrient imbalance remediation + improved soil health + increased household human and physical capital + increased landscape physical capital]')),
                                 Scenario,y=Probability
                                 ))+
  geom_boxplot()+
  scale_x_discrete(labels = label_wrap(14)) +
  scale_y_continuous(labels = comma) 

alli_p + geom_hline(yintercept=0.33,linetype="dashed", color = "black")+
  xlab('Intervention strategies')+ 
  ylab('Probabilty of low productivity')

 






##---Post-intervention outcomes------------------------------------------------------
#Post K_PNB intervention
K<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/outcome_k_pnb.csv', sep=""))
dim(K)
t<-569500
K$Scenario<-'K PNB intervention'
K<-K[c('Scenario',"Class")]
names(K)[2]<-'Probability'
colnames(K)
median(K$Probability)
library('dplyr')
K<-K%>%count(Probability)
K$percent<-(K$n/t)*100
K

#Post P_PNB intervention
P<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/outcome_p_pnb.csv', sep=""))
P$Scenario<-'P PNB intervention'
P<-P[c('Scenario',"Class")]
names(P)[2]<-'Probability'
colnames(P)
median(P$Probability)
P<-P%>%count(Probability)
P$percent<-(P$n/t)*100
P

#Post N_PNB intervention
N<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/outcome_n_pnb.csv', sep=""))
N$Scenario<-'N PNB intervention'
N<-N[c('Scenario',"Class")]
names(N)[2]<-'Probability'
colnames(N)
mean(N$Probability)
N<-N%>%count(Probability)
N$percent<-(N$n/t)*100
N

#Post Nutrient management intervention
NM<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/outcome_fert.csv', sep=""))
NM$Scenario<-'Nutrient management intervention'
NM<-NM[c('Scenario',"Class")]
names(NM)[2]<-'Probability'
colnames(NM)
mean(NM$Probability)
NM<-NM%>%count(Probability)
NM$percent<-(NM$n/t)*100
NM

#Labor intervention
L<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/outcome_labor.csv', sep=""))
L$Scenario<-'Labor intervention'
L<-L[c('Scenario',"Class")]
names(L)[2]<-'Probability'
colnames(L)
mean(L$Probability)
L<-L%>%count(Probability)
L$percent<-(L$n/t)*100
L

#Dependency intervention
D<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/outcome_hh_dep.csv', sep=""))
D$Scenario<-'Household dependency intervention'
D<-D[c('Scenario',"Class")]
names(D)[2]<-'Probability'
colnames(D)
mean(D$Probability)
D<-D%>%count(Probability)
D$percent<-(D$n/t)*100
D


#Households intervention
HH<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/outcome_hh.csv', sep=""))
HH$Scenario<-'Household interventions'
HH<-HH[c('Scenario',"Class")]
names(HH)[2]<-'Probability'
colnames(HH)
mean(HH$Probability)
HH<-HH%>%count(Probability)
HH$percent<-(HH$n/t)*100
HH

#Slope intervention
S<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/outcome_slope.csv', sep=""))
S$Scenario<-'Soil slope intervention'
S<-S[c('Scenario',"Class")]
names(S)[2]<-'Probability'
colnames(S)
mean(S$Probability)
S<-S%>%count(Probability)
S$percent<-(S$n/t)*100
S

#River intervention
R<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/outcome_dist_riv.csv', sep=""))
R$Scenario<-'River intervention'
R<-R[c('Scenario',"Class")]
names(R)[2]<-'Probability'
colnames(R)
mean(R$Probability)
R<-R%>%count(Probability)
R$percent<-(R$n/t)*100
R
#Roads intervention
Rd<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/outcome_dist_roads.csv', sep=""))
Rd$Scenario<-'Road intervention'
Rd<-Rd[c('Scenario',"Class")]
names(Rd)[2]<-'Probability'
colnames(Rd)
mean(Rd$Probability)
Rd<-Rd%>%count(Probability)
Rd$percent<-(Rd$n/t)*100
Rd

#landscape interventions
lands<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/outcome_lands.csv', sep=""))
lands$Scenario<-'Road intervention'
lands<-lands[c('Scenario',"Class")]
names(lands)[2]<-'Probability'
colnames(lands)
mean(lands$Probability)
lands<-lands%>%count(Probability)
lands$percent<-(lands$n/t)*100
lands

#All interventions
all<- read.csv(paste('./8_Analyses/ML_GSA/post_int_prob/outcome_all.csv', sep=""))
all$Scenario<-'Combined interventions'
all<-all[c('Scenario',"Class")]
names(all)[2]<-'Probability'
colnames(all)
mean(S$Probability)
all<-all%>%count(Probability)
all$percent<-(all$n/t)*100
all



