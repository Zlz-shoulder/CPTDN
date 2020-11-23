library(survcomp)
library(survival)
#read matched data
wb = 'matched_data_from_center1.csv'
clinic_data =  read.csv(wb, header = TRUE, sep = ',')
row.names(clinic_data) = clinic_data$Pat_ID
tra_data = subset(clinic_data, data_cohort2==0)
test_data = subset(clinic_data, data_cohort2==1)

wb = 'matched_data_from_center2-4.csv'
match_ext_data =  read.csv(wb, header = TRUE, sep = ',')
row.names(match_ext_data) = match_ext_data$Pat_ID

tra_data$treatment = ifelse(tra_data$treatment>0,1,0)
test_data$treatment = ifelse(test_data$treatment>0,1,0)
match_ext_data$treatment = ifelse(match_ext_data$treatment>0,1,0)

tra_data$DFS = ifelse(tra_data$DFS>0,1,0)
test_data$DFS = ifelse(test_data$DFS>0,1,0)
match_ext_data$DFS = ifelse(match_ext_data$DFS>0,1,0)

tra_data$OS = ifelse(tra_data$OS>0,1,0)
test_data$OS = ifelse(test_data$OS>0,1,0)
match_ext_data$OS = ifelse(match_ext_data$OS>0,1,0)

tra_data$DMFS = ifelse(tra_data$DMFS>0,1,0)
test_data$DMFS = ifelse(test_data$DMFS>0,1,0)
match_ext_data$DMFS = ifelse(match_ext_data$DMFS>0,1,0)

tra_data$LRRFS = ifelse(tra_data$LRRFS>0,1,0)
test_data$LRRFS = ifelse(test_data$LRRFS>0,1,0)
match_ext_data$LRRFS = ifelse(match_ext_data$LRRFS>0,1,0)
#calculate follow-up time
summary(tra_data$OS.time/30)
summary(test_data$OS.time/30)
summary(match_ext_data$OS.time/30)

#Baseline characteristics in three cohorts (Table 1).
#Continuous variables were tested by either the Kruskal-Wallis rank sum test
summary(tra_data$age)
summary(test_data$age)
summary(match_ext_data$age)
x1 = c(tra_data$age, test_data$age, match_ext_data$age)
group = c(rep(1,nrow(tra_data)),rep(2,nrow(test_data)),rep(3,nrow(match_ext_data)))
kruskal.test(x1,factor(group))

summary(tra_data$tumor_volume)
summary(test_data$tumor_volume)
summary(match_ext_data$tumor_volume)
x1 = c(tra_data$tumor_volume, test_data$tumor_volume, match_ext_data$tumor_volume)
group = c(rep(1,nrow(tra_data)),rep(2,nrow(test_data)),rep(3,nrow(match_ext_data)))
kruskal.test(x1,factor(group))

#categorical variables were tested by either the Pearson??беs |??2 test or the Fisher??беs exact test.
source('chisq-fisher-test.r')
name = c("DFS","sex","HGBcut","CRPcut","LDHcut","ALBcut","EBV_4k","smokingcut","WHOcut","drinkingcut","His_cancercut","treatment")
for(s in name){
  var = c(tra_data[,s], test_data[,s])
  x1 = match_ext_data[,s]
  mark = is.na(x1)
  print("No of the loss:")
  print(sum(mark))
  x1 = x1[!mark]
  var = c(var, x1)
  group = c(rep(1,684), rep(2,324), rep(3,length(x1)))
  tmp = data.frame(group = group)
  tmp$var = var
  tab0 = xtabs(~group+var, data = tmp)
  print(s)
  print(tab0)
  print(prop.table(tab0,1))
  res = get.test.method(tab0)
  print("p value:")
  print(res)
  print("******************")
}

#read deep-learning signature
wb = 'all_MR_dlFeature.csv'
DL_data =  read.csv(wb, header = TRUE, sep = ',')
row.names(DL_data) = DL_data$Pat_ID

name1 = row.names(tra_data)
tra_data$Prog_score = DL_data[name1, 'Prog_score']
tra_data$Pred_score = DL_data[name1, 'Pred_score']


name1 = row.names(test_data)
test_data$Prog_score = DL_data[name1, 'Prog_score']
test_data$Pred_score = DL_data[name1, 'Pred_score']

name1 = row.names(match_ext_data)
match_ext_data$Prog_score = DL_data[name1, 'Prog_score']
match_ext_data$Pred_score = DL_data[name1, 'Pred_score']

tra_data$Pred_scorecut = ifelse(tra_data$Pred_score>0,1,0)
tra_data$Pred_scorecut = ifelse(tra_data$Pred_score>0,1,0)
tra_data$Pred_scorecut = ifelse(tra_data$Pred_score>0,1,0)

test_data$Pred_scorecut = ifelse(test_data$Pred_score>0,1,0)
test_data$Pred_scorecut = ifelse(test_data$Pred_score>0,1,0)
test_data$Pred_scorecut = ifelse(test_data$Pred_score>0,1,0)

match_ext_data$Pred_scorecut = ifelse(match_ext_data$Pred_score>0,1,0)
match_ext_data$Pred_scorecut = ifelse(match_ext_data$Pred_score>0,1,0)
match_ext_data$Pred_scorecut = ifelse(match_ext_data$Pred_score>0,1,0)


#Performance of Prognostic-score in three cohorts
cd = concordance.index(x = tra_data$Prog_score, surv.time=tra_data$DFS.time, surv.event=tra_data$DFS,method = "noether")
cd[1:6]

cd = concordance.index(x = test_data$Prog_score, surv.time=test_data$DFS.time, surv.event=test_data$DFS,method = "noether")
cd[1:6]

cd = concordance.index(x = match_ext_data$Prog_score, surv.time=match_ext_data$DFS.time, surv.event=match_ext_data$DFS,method = "noether")
cd[1:6]


#Assocication of Predictive-score with treatment in three cohorts
coxph(Surv(DFS.time,DFS) ~ Pred_score*treatment, data = tra_data)

coxph(Surv(DFS.time,DFS) ~ Pred_score*treatment, data = test_data)

coxph(Surv(DFS.time,DFS) ~ Pred_score*treatment, data = match_ext_data)

#Association of baseline characteristics with Predictive-score (Table D1).
name = c("sex","HGBcut","CRPcut","LDHcut","ALBcut","EBV_4k","smokingcut","drinkingcut","His_cancercut",'age','tumor_volume','Pred_scorecut')
tmp_data = tra_data[,name]
x1 = tmp_data$tumor_volume
group = tmp_data$Pred_scorecut
kruskal.test(x1,factor(group))
summary(tmp_data$tumor_volume[tmp_data$Pred_scorecut==0])
summary(tmp_data$tumor_volume[tmp_data$Pred_scorecut==1])

x1 = tmp_data$age
group = tmp_data$Pred_scorecut
kruskal.test(x1,factor(group))
summary(tmp_data$age[tmp_data$Pred_scorecut==0])
summary(tmp_data$age[tmp_data$Pred_scorecut==1])

name =  c("sex","HGBcut","CRPcut","LDHcut","ALBcut","EBV_4k","smokingcut","drinkingcut","His_cancercut")
for(s in name){
  x1 = tmp_data[,s]
  mark = is.na(x1)
  print("No of the loss:")
  print(sum(mark))
  var = x1[!mark]
  group = tmp_data$Pred_scorecut[!mark]
  tmp = data.frame(group = group)
  tmp$var = var
  tab0 = xtabs(~group+var, data = tmp)
  print(s)
  print(tab0)
  print(prop.table(tab0,1))
  res = get.test.method(tab0)
  print("p value:")
  print(res)
  print("******************")
}


#internal test cohort
name = c("sex","HGBcut","CRPcut","LDHcut","ALBcut","EBV_4k","smokingcut","drinkingcut","His_cancercut",'age','tumor_volume','Pred_scorecut')
tmp_data = test_data[,name]
x1 = tmp_data$tumor_volume
group = tmp_data$Pred_scorecut
kruskal.test(x1,factor(group))
summary(tmp_data$tumor_volume[tmp_data$Pred_scorecut==0])
summary(tmp_data$tumor_volume[tmp_data$Pred_scorecut==1])

x1 = tmp_data$age
group = tmp_data$Pred_scorecut
kruskal.test(x1,factor(group))
summary(tmp_data$age[tmp_data$Pred_scorecut==0])
summary(tmp_data$age[tmp_data$Pred_scorecut==1])

name =  c("sex","HGBcut","CRPcut","LDHcut","ALBcut","EBV_4k","smokingcut","drinkingcut","His_cancercut")
for(s in name){
  x1 = tmp_data[,s]
  mark = is.na(x1)
  print("No of the loss:")
  print(sum(mark))
  var = x1[!mark]
  group = tmp_data$Pred_scorecut[!mark]
  tmp = data.frame(group = group)
  tmp$var = var
  tab0 = xtabs(~group+var, data = tmp)
  print(s)
  print(tab0)
  print(prop.table(tab0,1))
  res = get.test.method(tab0)
  print("p value:")
  print(res)
  print("******************")
}

#external test cohort
name = c("sex","HGBcut","CRPcut","LDHcut","ALBcut","EBV_4k","smokingcut","drinkingcut","His_cancercut",'age','tumor_volume','Pred_scorecut')
tmp_data = match_ext_data[,name]
x1 = tmp_data$tumor_volume
group = tmp_data$Pred_scorecut
kruskal.test(x1,factor(group))
summary(tmp_data$tumor_volume[tmp_data$Pred_scorecut==0])
summary(tmp_data$tumor_volume[tmp_data$Pred_scorecut==1])

x1 = tmp_data$age
group = tmp_data$Pred_scorecut
kruskal.test(x1,factor(group))
summary(tmp_data$age[tmp_data$Pred_scorecut==0])
summary(tmp_data$age[tmp_data$Pred_scorecut==1])

name =  c("sex","HGBcut","CRPcut","LDHcut","ALBcut","EBV_4k","smokingcut","drinkingcut","His_cancercut")
for(s in name){
  x1 = tmp_data[,s]
  mark = is.na(x1)
  print("No of the loss:")
  print(sum(mark))
  var = x1[!mark]
  group = tmp_data$Pred_scorecut[!mark]
  tmp = data.frame(group = group)
  tmp$var = var
  tab0 = xtabs(~group+var, data = tmp)
  print(s)
  print(tab0)
  print(prop.table(tab0,1))
  res = get.test.method(tab0)
  print("p value:")
  print(res)
  print("******************")
}

# Univariate Cox proportional hazards regression analysis for clinical factors
library(forestplot)
clinic_ind = c('age','sex','smokingcut','drinkingcut','His_cancercut',
               'HGBcut','ALBcut','CRPcut','LDHcut','EBV_4k','tumor_volume')
ft_plot = data.frame(factor = clinic_ind)
row.names(ft_plot) = clinic_ind
ft_plot$Point.Estimate =0
ft_plot$Low = 0
ft_plot$High = 0
ft_plot$p.value = 1.0
ft_plot$p.interaction = 1.0
for(s in clinic_ind){
  h = hazard.ratio(x = tra_data[,s], surv.time = tra_data$DFS.time, surv.event = tra_data$DFS)
  ft_plot[s, 'Point.Estimate'] = h$hazard.ratio
  ft_plot[s, 'Low'] = h$lower
  ft_plot[s, 'High'] = h$upper
  ft_plot[s, 'p.value'] = h$p.value
  cp <- coxph(as.formula(paste("Surv(DFS.time,DFS) ~ ", paste(s,"*treatment"))), data = tra_data)
  sa <- summary(cp)
  ft_plot[s, 'p.interaction'] = round(sa$coefficients[3,5],2)
  
}

clinic_ind = c('age','tumor_volume')
for(s in clinic_ind){
  h = hazard.ratio(x = 10*tra_data[,s], surv.time = tra_data$DFS.time, surv.event = tra_data$DFS)
  ft_plot[s, 'Point.Estimate'] = h$hazard.ratio
  ft_plot[s, 'Low'] = h$lower
  ft_plot[s, 'High'] = h$upper
  ft_plot[s, 'p.value'] = h$p.value
  cp <- coxph(as.formula(paste("Surv(DFS.time,DFS) ~ ", paste(s,"*treatment"))), data = tra_data)
  sa <- summary(cp)
  ft_plot[s, 'p.interaction'] = round(sa$coefficients[3,5],2)
  
}

ft_plot$hr_CI = paste(as.character(round(ft_plot$Point.Estimate,4)), " (", as.character(round(ft_plot$Low,4)),
                      "-", as.character(round(ft_plot$High,4)), ")", sep = "")

tabletext <- cbind(c("Factor",as.character(ft_plot$factor)), c("P-interaction",as.character(ft_plot$p.interaction)),
                   c("Hazard Ratio",ft_plot$hr_CI), c("P-Value",round(ft_plot$p.value,4)))


dev.new()
forestplot(labeltext=tabletext, graph.pos=4,
           mean=c(NA,ft_plot$Point.Estimate),
           lower=c(NA,ft_plot$Low), upper=c(NA,ft_plot$High),
           clip=c(0.1,10),
           col=fpColors(box="#1c61b6", lines="#1c61b6", zero = "gray50"),
           zero=1, cex=0.9, lineheight = "auto", boxsize=0.2, colgap=unit(5,"mm"),
           lwd.ci=2, ci.vertices=F, ci.vertices.height = 0.2)

#Model construction
tra_CCRT_data = subset(tra_data, treatment == 0)
tra_ICT_data = subset(tra_data, treatment == 1)
test_CCRT_data = subset(test_data, treatment == 0)
test_ICT_data = subset(test_data, treatment == 1)

flag = is.na(match_ext_data$EBV_4k)
match_ext_data_EBV = match_ext_data[!flag,]
ext_EBV_CCRT = subset(match_ext_data_EBV, treatment == 0)
ext_EBV_ICT = subset(match_ext_data_EBV, treatment == 1)

model_performance = c()
#Training the model in patients receiving CCRT alone using clinical factors
model_CCRT_clinic <- coxph(Surv(DFS.time,DFS) ~smokingcut+drinkingcut+age+tumor_volume+sex+EBV_4k+HGBcut+ALBcut+LDHcut+CRPcut, data = tra_CCRT_data)
summary(model_CCRT_clinic)
model_CCRT_clinic = step(model_CCRT_clinic)
summary(model_CCRT_clinic)
tmp = c()

#Training cohort
tra1 <- predict(model_CCRT_clinic, newdata = tra_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=tra_data$DFS.time, surv.event=tra_data$DFS,method = "noether")
cd[1:6]


tra1 <- predict(model_CCRT_clinic, newdata = tra_CCRT_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=tra_CCRT_data$DFS.time, surv.event=tra_CCRT_data$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

tra1 <- predict(model_CCRT_clinic, newdata = tra_ICT_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=tra_ICT_data$DFS.time, surv.event=tra_ICT_data$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

#Internal test cohort
tra1 <- predict(model_CCRT_clinic, newdata = test_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=test_data$DFS.time, surv.event=test_data$DFS,method = "noether")
cd[1:6]

tra1 <- predict(model_CCRT_clinic, newdata = test_CCRT_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=test_CCRT_data$DFS.time, surv.event=test_CCRT_data$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

tra1 <- predict(model_CCRT_clinic, newdata = test_ICT_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=test_ICT_data$DFS.time, surv.event=test_ICT_data$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)


#External test cohort
tra1 <- predict(model_CCRT_clinic, newdata = match_ext_data_EBV,type = "lp")
cd = concordance.index(x = tra1, surv.time=match_ext_data_EBV$DFS.time, surv.event=match_ext_data_EBV$DFS,method = "noether")
cd[1:6]

tra1 <- predict(model_CCRT_clinic, newdata = ext_EBV_CCRT,type = "lp")
cd = concordance.index(x = tra1, surv.time=ext_EBV_CCRT$DFS.time, surv.event=ext_EBV_CCRT$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

tra1 <- predict(model_CCRT_clinic, newdata = ext_EBV_ICT,type = "lp")
cd = concordance.index(x = tra1, surv.time=ext_EBV_ICT$DFS.time, surv.event=ext_EBV_ICT$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

model_performance = c(model_performance, tmp)

#Training the model in patients receiving ICT+CCRT using clinical factors
model_ICT_clinic <- coxph(Surv(DFS.time,DFS) ~ smokingcut+drinkingcut+age+tumor_volume+sex+EBV_4k+HGBcut+ALBcut+LDHcut+CRPcut+smokingcut, data = tra_ICT_data)
summary(model_ICT_clinic)
model_ICT_clinic = step(model_ICT_clinic)
summary(model_ICT_clinic)
tmp = c()

#Training cohort
tra1 <- predict(model_ICT_clinic, newdata = tra_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=tra_data$DFS.time, surv.event=tra_data$DFS,method = "noether")
cd[1:6]


tra1 <- predict(model_ICT_clinic, newdata = tra_CCRT_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=tra_CCRT_data$DFS.time, surv.event=tra_CCRT_data$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

tra1 <- predict(model_ICT_clinic, newdata = tra_ICT_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=tra_ICT_data$DFS.time, surv.event=tra_ICT_data$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)
############
#Internal test cohort
tra1 <- predict(model_ICT_clinic, newdata = test_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=test_data$DFS.time, surv.event=test_data$DFS,method = "noether")
cd[1:6]

tra1 <- predict(model_ICT_clinic, newdata = test_CCRT_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=test_CCRT_data$DFS.time, surv.event=test_CCRT_data$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

tra1 <- predict(model_ICT_clinic, newdata = test_ICT_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=test_ICT_data$DFS.time, surv.event=test_ICT_data$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)


#external test cohort
tra1 <- predict(model_ICT_clinic, newdata = match_ext_data_EBV,type = "lp")
cd = concordance.index(x = tra1, surv.time=match_ext_data_EBV$DFS.time, surv.event=match_ext_data_EBV$DFS,method = "noether")
cd[1:6]

tra1 <- predict(model_ICT_clinic, newdata = ext_EBV_CCRT,type = "lp")
cd = concordance.index(x = tra1, surv.time=ext_EBV_CCRT$DFS.time, surv.event=ext_EBV_CCRT$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

tra1 <- predict(model_ICT_clinic, newdata = ext_EBV_ICT,type = "lp")
cd = concordance.index(x = tra1, surv.time=ext_EBV_ICT$DFS.time, surv.event=ext_EBV_ICT$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

model_performance = c(model_performance, tmp)


#Training the model in the entire patients regardless of treatment regimen using clinical factors
model_ICT.CCRT_clinic <- coxph(Surv(DFS.time,DFS) ~ smokingcut+drinkingcut+age+tumor_volume+sex+EBV_4k+HGBcut+ALBcut+LDHcut+CRPcut, data = tra_data)
summary(model_ICT.CCRT_clinic)
model_ICT.CCRT_clinic = step(model_ICT.CCRT_clinic)
summary(model_ICT.CCRT_clinic)
tmp = c()

tra1 <- predict(model_ICT.CCRT_clinic, newdata = tra_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=tra_data$DFS.time, surv.event=tra_data$DFS,method = "noether")
cd[1:6]


tra1 <- predict(model_ICT.CCRT_clinic, newdata = tra_CCRT_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=tra_CCRT_data$DFS.time, surv.event=tra_CCRT_data$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

tra1 <- predict(model_ICT.CCRT_clinic, newdata = tra_ICT_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=tra_ICT_data$DFS.time, surv.event=tra_ICT_data$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

#Internal test cohort
tra1 <- predict(model_ICT.CCRT_clinic, newdata = test_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=test_data$DFS.time, surv.event=test_data$DFS,method = "noether")
cd[1:6]

tra1 <- predict(model_ICT.CCRT_clinic, newdata = test_CCRT_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=test_CCRT_data$DFS.time, surv.event=test_CCRT_data$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

tra1 <- predict(model_ICT.CCRT_clinic, newdata = test_ICT_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=test_ICT_data$DFS.time, surv.event=test_ICT_data$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)


#External test cohort
tra1 <- predict(model_ICT.CCRT_clinic, newdata = match_ext_data_EBV,type = "lp")
cd = concordance.index(x = tra1, surv.time=match_ext_data_EBV$DFS.time, surv.event=match_ext_data_EBV$DFS,method = "noether")
cd[1:6]

tra1 <- predict(model_ICT.CCRT_clinic, newdata = ext_EBV_CCRT,type = "lp")
cd = concordance.index(x = tra1, surv.time=ext_EBV_CCRT$DFS.time, surv.event=ext_EBV_CCRT$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

tra1 <- predict(model_ICT.CCRT_clinic, newdata = ext_EBV_ICT,type = "lp")
cd = concordance.index(x = tra1, surv.time=ext_EBV_ICT$DFS.time, surv.event=ext_EBV_ICT$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

model_performance = c(model_performance, tmp)


########################
#Performance of Prognostic-score
model_ICT.CCRT_rad <- coxph(Surv(DFS.time,DFS) ~ Prog_score, data = tra_data)
tmp = c()

tra1 <- predict(model_ICT.CCRT_rad, newdata = tra_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=tra_data$DFS.time, surv.event=tra_data$DFS,method = "noether")
cd[1:6]


tra1 <- predict(model_ICT.CCRT_rad, newdata = tra_CCRT_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=tra_CCRT_data$DFS.time, surv.event=tra_CCRT_data$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

tra1 <- predict(model_ICT.CCRT_rad, newdata = tra_ICT_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=tra_ICT_data$DFS.time, surv.event=tra_ICT_data$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)
############
#Internal test cohort
tra1 <- predict(model_ICT.CCRT_rad, newdata = test_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=test_data$DFS.time, surv.event=test_data$DFS,method = "noether")
cd[1:6]

tra1 <- predict(model_ICT.CCRT_rad, newdata = test_CCRT_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=test_CCRT_data$DFS.time, surv.event=test_CCRT_data$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

tra1 <- predict(model_ICT.CCRT_rad, newdata = test_ICT_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=test_ICT_data$DFS.time, surv.event=test_ICT_data$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

match_ext_data_CCRT = subset(match_ext_data, treatment == 0)
match_ext_data_ICT = subset(match_ext_data, treatment == 1)
#External test cohort
tra1 <- predict(model_ICT.CCRT_rad, newdata = match_ext_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=match_ext_data$DFS.time, surv.event=match_ext_data$DFS,method = "noether")
cd[1:6]

tra1 <- predict(model_ICT.CCRT_rad, newdata = match_ext_data_CCRT,type = "lp")
cd = concordance.index(x = tra1, surv.time=match_ext_data_CCRT$DFS.time, surv.event=match_ext_data_CCRT$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

tra1 <- predict(model_ICT.CCRT_rad, newdata = match_ext_data_ICT,type = "lp")
cd = concordance.index(x = tra1, surv.time=match_ext_data_ICT$DFS.time, surv.event=match_ext_data_ICT$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

model_performance = c(model_performance, tmp)


#Training the model in the entire patients using clinical factors and Prognostic-score
tmp = c()
model_ICT.CCRT_clinic.rad <- coxph(Surv(DFS.time,DFS) ~ age+Prog_score+EBV_4k, data = tra_data)
summary(model_ICT.CCRT_clinic.rad)

tra1 <- predict(model_ICT.CCRT_clinic.rad, newdata = tra_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=tra_data$DFS.time, surv.event=tra_data$DFS,method = "noether")
cd[1:6]


tra1 <- predict(model_ICT.CCRT_clinic.rad, newdata = tra_CCRT_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=tra_CCRT_data$DFS.time, surv.event=tra_CCRT_data$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

tra1 <- predict(model_ICT.CCRT_clinic.rad, newdata = tra_ICT_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=tra_ICT_data$DFS.time, surv.event=tra_ICT_data$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)
############
#Internal test cohort
tra1 <- predict(model_ICT.CCRT_clinic.rad, newdata = test_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=test_data$DFS.time, surv.event=test_data$DFS,method = "noether")
cd[1:6]

tra1 <- predict(model_ICT.CCRT_clinic.rad, newdata = test_CCRT_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=test_CCRT_data$DFS.time, surv.event=test_CCRT_data$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

tra1 <- predict(model_ICT.CCRT_clinic.rad, newdata = test_ICT_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=test_ICT_data$DFS.time, surv.event=test_ICT_data$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)


#external test cohort
tra1 <- predict(model_ICT.CCRT_clinic.rad, newdata = match_ext_data_EBV,type = "lp")
cd = concordance.index(x = tra1, surv.time=match_ext_data_EBV$DFS.time, surv.event=match_ext_data_EBV$DFS,method = "noether")
cd[1:6]

tra1 <- predict(model_ICT.CCRT_clinic.rad, newdata = ext_EBV_CCRT,type = "lp")
cd = concordance.index(x = tra1, surv.time=ext_EBV_CCRT$DFS.time, surv.event=ext_EBV_CCRT$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

tra1 <- predict(model_ICT.CCRT_clinic.rad, newdata = ext_EBV_ICT,type = "lp")
cd = concordance.index(x = tra1, surv.time=ext_EBV_ICT$DFS.time, surv.event=ext_EBV_ICT$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)


model_performance = c(model_performance, tmp)


#Training the model (CPTDN) in the entire patients using clinical factors, Prognostic-score, and interaction item of treatment with Predictive-score
model_ICT.CCRT_clinic.rad.Pred_scorecut <- coxph(Surv(DFS.time,DFS) ~ age+Prog_score+Pred_score*treatment+EBV_4k, data = tra_data)
summary(model_ICT.CCRT_clinic.rad.Pred_scorecut)
tmp = c()

tra1 <- predict(model_ICT.CCRT_clinic.rad.Pred_scorecut, newdata = tra_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=tra_data$DFS.time, surv.event=tra_data$DFS,method = "noether")
cd[1:6]


tra1 <- predict(model_ICT.CCRT_clinic.rad.Pred_scorecut, newdata = tra_CCRT_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=tra_CCRT_data$DFS.time, surv.event=tra_CCRT_data$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

tra1 <- predict(model_ICT.CCRT_clinic.rad.Pred_scorecut, newdata = tra_ICT_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=tra_ICT_data$DFS.time, surv.event=tra_ICT_data$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)
############
#Internal test cohort
tra1 <- predict(model_ICT.CCRT_clinic.rad.Pred_scorecut, newdata = test_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=test_data$DFS.time, surv.event=test_data$DFS,method = "noether")
cd[1:6]

tra1 <- predict(model_ICT.CCRT_clinic.rad.Pred_scorecut, newdata = test_CCRT_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=test_CCRT_data$DFS.time, surv.event=test_CCRT_data$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

tra1 <- predict(model_ICT.CCRT_clinic.rad.Pred_scorecut, newdata = test_ICT_data,type = "lp")
cd = concordance.index(x = tra1, surv.time=test_ICT_data$DFS.time, surv.event=test_ICT_data$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)


#external test cohort
tra1 <- predict(model_ICT.CCRT_clinic.rad.Pred_scorecut, newdata = match_ext_data_EBV,type = "lp")
cd = concordance.index(x = tra1, surv.time=match_ext_data_EBV$DFS.time, surv.event=match_ext_data_EBV$DFS,method = "noether")
cd[1:6]

tra1 <- predict(model_ICT.CCRT_clinic.rad.Pred_scorecut, newdata = ext_EBV_CCRT,type = "lp")
cd = concordance.index(x = tra1, surv.time=ext_EBV_CCRT$DFS.time, surv.event=ext_EBV_CCRT$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

tra1 <- predict(model_ICT.CCRT_clinic.rad.Pred_scorecut, newdata = ext_EBV_ICT,type = "lp")
cd = concordance.index(x = tra1, surv.time=ext_EBV_ICT$DFS.time, surv.event=ext_EBV_ICT$DFS,method = "noether")
cd[1:6]
tmp = c(tmp,cd$c.index,cd$lower,cd$upper)

model_performance = c(model_performance, tmp)


#Comparison of prognostic models??бе performance on different treatment-specific subgroups (Figure 2)

nn = length(model_performance)
data_esti_CCRT = model_performance[seq(1,nn,6)]
lower_esti_CCRT = model_performance[seq(2,nn,6)]
upper_esti_CCRT = model_performance[seq(3,nn,6)]
esti_CCRT_matrix = matrix(data_esti_CCRT, nrow = 6, ncol = 3, byrow = T)
lower_CCRT_matrix = matrix(lower_esti_CCRT, nrow = 6, ncol = 3, byrow = T)
upper_CCRT_matrix = matrix(upper_esti_CCRT, nrow = 6, ncol = 3, byrow = T)


data_esti_ICT = model_performance[seq(4,nn,6)]
lower_esti_ICT = model_performance[seq(5,nn,6)]
upper_esti_ICT = model_performance[seq(6,nn,6)]
esti_ICT_matrix = matrix(data_esti_ICT, nrow = 6, ncol = 3, byrow = T)
lower_ICT_matrix = matrix(lower_esti_ICT, nrow = 6, ncol = 3, byrow = T)
upper_ICT_matrix = matrix(upper_esti_ICT, nrow = 6, ncol = 3, byrow = T)

data_esti = cbind(esti_CCRT_matrix, esti_ICT_matrix) 
colnames(data_esti) = c('Tra_CCRT','IntVal_CCRT','ExtVal_CCRT','Tra_ICT','IntVal_ICT','ExtVal_ICT')
row.names(data_esti) = c('model_CCRT_clinic','model_ICT_clinic','model_ICT+CCRT_clinic',
                         'model_ICT+CCRT_rad','model_ICT+CCRT_clinic+rad','model_ICT+CCRT_clinic+rad+Pred_scorecut')
bb = as.matrix(data_esti)
bb = round(bb,3)
# bb = t(bb)
lower_esti = cbind(lower_CCRT_matrix, lower_ICT_matrix)
upper_esti = cbind(upper_CCRT_matrix, upper_ICT_matrix)

lower_esti = as.data.frame(lower_esti)
upper_esti = as.data.frame(upper_esti)
colnames(lower_esti) = c('Tra_CCRT','IntVal_CCRT','ExtVal_CCRT','Tra_ICT','IntVal_ICT','ExtVal_ICT')
colnames(upper_esti) = c('Tra_CCRT','IntVal_CCRT','ExtVal_CCRT','Tra_ICT','IntVal_ICT','ExtVal_ICT')

row.names(lower_esti) = c('model_CCRT_clinic','model_ICT_clinic','model_ICT+CCRT_clinic',
                          'model_ICT+CCRT_rad','model_ICT+CCRT_clinic+rad','model_ICT+CCRT_clinic+rad+Pred_scorecut')
row.names(upper_esti) = c('model_CCRT_clinic','model_ICT_clinic','model_ICT+CCRT_clinic',
                          'model_ICT+CCRT_rad','model_ICT+CCRT_clinic+rad','model_ICT+CCRT_clinic+rad+Pred_scorecut')
lower_esti = as.matrix(lower_esti)
upper_esti = as.matrix(upper_esti)
library(gplots)
# dev.new()
cl_index = c(0.6509803921568628, 0.807843137254902, 0.8901960784313725,
             0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
             0.6980392156862745, 0.8745098039215686, 0.5411764705882353,
             0.2, 0.6274509803921569, 0.17254901960784313,
             0.984313725490196, 0.6039215686274509, 0.6,
             0.8901960784313725, 0.10196078431372549, 0.10980392156862745)
cl_index = matrix(cl_index, nrow = 6, ncol = 3, byrow = T)
barplot2(bb,beside=T,plot.grid = TRUE,plot.ci=TRUE, ylim=c(0.2,1.0),ci.lwd=2,cex.names =1.0,ci.l=lower_esti,ci.u=upper_esti,xpd=FALSE,
         col = rgb(cl_index, max = 1), border = "black")
box()
legend("topright", cex=1.2,legend=rownames(bb), fill =rgb(cl_index, max = 1),text.col = "black")

#Plot CPTDN and its calibration curves (Figure 3)
##nomogram
library(lattice);library(survival);library(Formula);library(ggplot2);library(Hmisc);library(rms)

ddist0 <- datadist(tra_data)
options(datadist='ddist0')
f <- cph(Surv(DFS.time,DFS) ~ Prog_score+age+EBV_4k+Pred_score*treatment, surv = TRUE, x = T, y = T, data = tra_data)
surv.prob <- Survival(f) # Construct survival probability function
nom <- nomogram(f, fun=list(function(x) surv.prob(1080, x),function(x) surv.prob(1800, x)),
                funlabel=c("3-year DFS rate","5-year DFS rate"),
                fun.at=c(0.95,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0),
                lp=F)
# plot(nom)
summary(nom)
tra1 <- predict(model_ICT.CCRT_clinic.rad.Pred_scorecut, newdata = tra_data,type = "lp")
tra_data$nomo_sig = tra1

############
#Internal test cohort
tra1 <- predict(model_ICT.CCRT_clinic.rad.Pred_scorecut, newdata = test_data,type = "lp")
test_data$nomo_sig = tra1

#external validation
tra1 <- predict(model_ICT.CCRT_clinic.rad.Pred_scorecut, newdata = match_ext_data_EBV,type = "lp")
match_ext_data_EBV$nomo_sig = tra1



cd = concordance.index(x = tra_data$nomo_sig, surv.time=tra_data$DFS.time, surv.event=tra_data$DFS,method = "noether")
cd[1:6]
cd = concordance.index(x = test_data$nomo_sig, surv.time=test_data$DFS.time, surv.event=test_data$DFS,method = "noether")
cd[1:6]
cd = concordance.index(x = match_ext_data_EBV$nomo_sig, surv.time=match_ext_data_EBV$DFS.time, surv.event=match_ext_data_EBV$DFS,method = "noether")
cd[1:6]

set.seed(321)
f_3 <- cph(Surv(DFS.time, DFS) ~ nomo_sig, surv = TRUE, x = T, y = T, data = tra_data
           ,time.inc = 1080)
cal_3 <- calibrate(f_3,  u=1080, cmethod='KM',method='crossvalidation', B = 100,m=220,surv=TRUE, time.inc=1080)

f_3 <- cph(Surv(DFS.time, DFS) ~ nomo_sig, surv = TRUE, x = T, y = T, data = test_data
           ,time.inc = 1080)
test_cal_3 <- calibrate(f_3,  u=1080, cmethod='KM',method='crossvalidation', B = 100,m=105,surv=TRUE, time.inc=1080)

f_3 <- cph(Surv(DFS.time, DFS) ~ nomo_sig, surv = TRUE, x = T, y = T, data = match_ext_data_EBV
           ,time.inc = 1080)
Ext_cal_3 <- calibrate(f_3,  u=1080, cmethod='KM',method='crossvalidation', B = 100,m=55,surv=TRUE, time.inc=1080)

source("HL-test.r")
y1 = HLtest(cal_3)
y1 = paste("Training: p =",as.character(round(y1,2)), sep = " ")
y2 = HLtest(test_cal_3)
y2 = paste("Internal test: p =",as.character(round(y2,2)), sep = " ")
y3 = HLtest(Ext_cal_3)
y3 = paste("External test: p =",as.character(round(y3,2)), sep = " ")






f_5 <- cph(Surv(DFS.time, DFS) ~ nomo_sig, surv = TRUE, x = T, y = T, data = tra_data
           ,time.inc = 1800)
cal_5 <- calibrate(f_5,  u=1800, cmethod='KM',method='crossvalidation', B = 100,m=220,surv=TRUE, time.inc=1800)

f_5 <- cph(Surv(DFS.time, DFS) ~ nomo_sig, surv = TRUE, x = T, y = T, data = test_data
           ,time.inc = 1800)
test_cal_5 <- calibrate(f_5,  u=1800, cmethod='KM',method='crossvalidation', B = 100,m=105,surv=TRUE, time.inc=1800)

f_5 <- cph(Surv(DFS.time, DFS) ~ nomo_sig, surv = TRUE, x = T, y = T, data = match_ext_data_EBV
           ,time.inc = 1800)
Ext_cal_5 <- calibrate(f_5,  u=1800, cmethod='KM',method='crossvalidation', B = 100,m=55,surv=TRUE, time.inc=1800)

source("HL-test.r")
y1_5 = HLtest(cal_5)
y1_5 = paste("Training: p =",as.character(round(y1_5,2)), sep = " ")
y2_5 = HLtest(test_cal_5)
y2_5 = paste("Internal test: p =",as.character(round(y2_5,2)), sep = " ")
y3_5 = HLtest(Ext_cal_5)
y3_5 = paste("External test: p =",as.character(round(y3_5,2)), sep = " ")

# dev.new()
#nomogram + calibration curves
opar <- par(no.readonly = TRUE)
par(mfrow = c(2,1))
par(lwd = 1, lty = 1)
x1 = c(0.01,0.99,0.45,0.99)
par(fig = x1)
plot(nom, xfrac=.3,cex.axis=1.3, cex.var=1.3)

x1 = c(0,0.5,0.02,0.56)
par(fig = x1,new = T)
par(lwd = 2, lty = 1)
x1 = "blue"
xm = 0.6
plot(cal_3,lty = 1,pch = 16,conf.int=F,xlim = c(xm,1),ylim = c(xm,1),riskdist = F,col = x1, axes = F)

abline(0, 1, lty = 5, col=c(rgb(220,220,220,maxColorValue = 255)) ,lwd=1)
x2 = "darkolivegreen3"
plot(test_cal_3,lty = 1,pch = 16,errbar.col = list(col=x2),xlim = c(xm,1),ylim = c(xm,1),col = x2,riskdist = F,add=T,conf.int=T)

x2 = "indianred1"
plot(Ext_cal_3,lty = 1,pch = 16,errbar.col = list(col=x2),xlim = c(xm,1),ylim = c(xm,1),col = x2,riskdist = F,add=T,conf.int=T)
axis(1,at=seq(xm,1.0,0.1),labels=seq(xm,1.0,0.1),pos=xm)
axis(2,at=seq(xm,1.0,0.1),labels=seq(xm,1.0,0.1),pos=xm)
legend("top", legend=c(y1,y2,y3),col=c('blue','darkolivegreen3','indianred1'), cex=1,lty=c(1,1,1))


x1 = c(0,0.5,0.02,0.56) + c(0.45,0.45,0,0)
par(fig = x1,new = T)
par(lwd = 2, lty = 1)
x1 = "blue"
xm = 0.6
plot(cal_5,lty = 1,pch = 16,conf.int=F,xlim = c(xm,1),ylim = c(xm,1),riskdist = F,col = x1, axes = F)

abline(0, 1, lty = 5, col=c(rgb(220,220,220,maxColorValue = 255)) ,lwd=1)
x2 = 'darkolivegreen3'
plot(test_cal_5,lty = 1,pch = 16,errbar.col = list(col=x2),xlim = c(xm,1),ylim = c(xm,1),col = x2,riskdist = F,add=T,conf.int=T)

x2 = "indianred1"
plot(Ext_cal_5,lty = 1,pch = 16,errbar.col = list(col=x2),xlim = c(xm,1),ylim = c(xm,1),col = x2,riskdist = F,add=T,conf.int=T)
axis(1,at=seq(xm,1.0,0.1),labels=seq(xm,1.0,0.1),pos=xm)
axis(2,at=seq(xm,1.0,0.1),labels=seq(xm,1.0,0.1),pos=xm)
legend("top", legend=c(y1_5,y2_5,y3_5),col=c('blue','darkolivegreen3','indianred1'), cex=1,lty=c(1,1,1))

par(opar)


########################################
#Prognostic performance of CPTDN (Figure 4)
#training cohort

summary(tra_data$nomo_sig)
cut_nomo = mean(tra_data$nomo_sig)

#calculate the total point corresponding to the mean of CPTDN outputs in the training cohort
factor(abs(tra_data$nomo_sig - cut_nomo)<0.002)
tra_data[242,'nomo_sig']
# tra_data[242,'dfs3']
name = c('EBV_4k','age','Prog_score', 'Pred_score','treatment','nomo_sig')
tra_data[242,name]

tra_data$nomo_sigcut = ifelse(tra_data$nomo_sig<cut_nomo,0,1)
test_data$nomo_sigcut = ifelse(test_data$nomo_sig<cut_nomo,0,1)
match_ext_data_EBV$nomo_sigcut = ifelse(match_ext_data_EBV$nomo_sig<cut_nomo,0,1)
tra_CCRT_data = subset(tra_data, treatment == 0)
tra_ICT_data = subset(tra_data, treatment == 1)
test_CCRT_data = subset(test_data, treatment == 0)
test_ICT_data = subset(test_data, treatment == 1)
ext_EBV_CCRT = subset(match_ext_data_EBV, treatment == 0)
ext_EBV_ICT = subset(match_ext_data_EBV, treatment == 1)
dev.new()
opar = par(no.readonly = T)
par(mfrow = c(3,3))
x1 = c(0,0.4,0.62,1.0)
par(fig = x1)
dd<-data.frame("surv.time" = tra_data$DFS.time, "surv.event" = tra_data$DFS,"strat" = tra_data$nomo_sigcut)
km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
              x.label="",y.label="Probability of survival",main="",.col = c('black','red'),
              leg.text=paste(c("low risk", "high risk")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
              , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))
x1 = x1+ c(0.3,0.3,0,0)
par(fig = x1, new = T)
dd<-data.frame("surv.time" = test_data$DFS.time, "surv.event" = test_data$DFS,"strat" = test_data$nomo_sigcut)
km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
              x.label="",y.label="",main="",.col = c('black','red'),
              leg.text=paste(c("low risk", "high risk")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
              , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))

x1 = x1+ c(0.3,0.3,0,0)
par(fig = x1, new = T)
dd<-data.frame("surv.time" = match_ext_data_EBV$DFS.time, "surv.event" = match_ext_data_EBV$DFS,"strat" = match_ext_data_EBV$nomo_sigcut)
km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
              x.label="",y.label="",main="",.col = c('black','red'),
              leg.text=paste(c("low risk", "high risk")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
              , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))



x1 = c(0,0.4,0.62,1.0) - c(0,0,0.3,0.3)
par(fig = x1, new = T)
dd<-data.frame("surv.time" = tra_CCRT_data$DFS.time, "surv.event" = tra_CCRT_data$DFS,"strat" = tra_CCRT_data$nomo_sigcut)
km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
              x.label="",main="",y.label="Probability of survival",.col = c('black','red'),
              leg.text=paste(c("low risk", "high risk")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
              , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))

x1 = x1+ c(0.3,0.3,0,0)
par(fig = x1, new = T)
dd<-data.frame("surv.time" = test_CCRT_data$DFS.time, "surv.event" = test_CCRT_data$DFS,"strat" = test_CCRT_data$nomo_sigcut)
km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
              x.label="",y.label="",main="",.col = c('black','red'),
              leg.text=paste(c("low risk", "high risk")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
              , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))
x1 = x1+ c(0.3,0.3,0,0)
par(fig = x1, new = T)
dd<-data.frame("surv.time" = ext_EBV_CCRT$DFS.time, "surv.event" = ext_EBV_CCRT$DFS,"strat" = ext_EBV_CCRT$nomo_sigcut)
km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
              x.label="",y.label="",main="",.col = c('black','red'),
              leg.text=paste(c("low risk", "high risk")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
              , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))



x1 = c(0,0.4,0.62,1.0) - c(0,0,0.6,0.6)
par(fig = x1, new = T)
dd<-data.frame("surv.time" = tra_ICT_data$DFS.time, "surv.event" = tra_ICT_data$DFS,"strat" = tra_ICT_data$nomo_sigcut)
km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
              x.label="Time (months)", y.label="Probability of survival",main="",.col = c('black','red'),
              leg.text=paste(c("low risk", "high risk")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
              , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))

x1 = x1+ c(0.3,0.3,0,0)
par(fig = x1, new = T)
dd<-data.frame("surv.time" = test_ICT_data$DFS.time, "surv.event" = test_ICT_data$DFS,"strat" = test_ICT_data$nomo_sigcut)
km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
              y.label="",main="",x.label="Time (months)" ,.col = c('black','red'),
              leg.text=paste(c("low risk", "high risk")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
              , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))
x1 = x1+ c(0.3,0.3,0,0)
par(fig = x1, new = T)
dd<-data.frame("surv.time" = ext_EBV_ICT$DFS.time, "surv.event" = ext_EBV_ICT$DFS,"strat" = ext_EBV_ICT$nomo_sigcut)
km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
              y.label="",main="",x.label="Time (months)", .col = c('black','red'),
              leg.text=paste(c("low risk", "high risk")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
              , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))
par(opar)


#Predictive performance of CPTDN (Figure 4)
#calculate the probability that a patient will not experience disease progression within 3 and 5 years
f <- cph(Surv(DFS.time,DFS) ~ age+EBV_4k+Prog_score+Pred_score*treatment, surv = TRUE, x = T, y = T, data = tra_data)
x = survest(f, newdata = tra_data,times=c(1080), conf.int=.95)
tra_data$dfs3 = x$surv
x = survest(f, newdata = tra_data,times=c(1800), conf.int=.95)
tra_data$dfs5 = x$surv
tmp_data = tra_data[,c('EBV_4k','age','Prog_score', 'Pred_score','nomo_sig')]
tmp_data$treatment = 1
x1 = survest(f, newdata = tmp_data,times=c(1800), conf.int=.95)
sur5.1 = x1$surv
tmp_data$treatment = -1
x2 = survest(f, newdata = tmp_data,times=c(1800), conf.int=.95)
sur5.2 = x2$surv
tra_data$delta5 = sur5.1-sur5.2

#internal test cohort
x = survest(f, newdata = test_data,times=c(1080), conf.int=.95)
test_data$dfs3 = x$surv
x = survest(f, newdata = test_data,times=c(1800), conf.int=.95)
test_data$dfs5 = x$surv
tmp_data = test_data[,c('EBV_4k','age','Prog_score', 'Pred_score','nomo_sig')]
tmp_data$treatment = 1
x1 = survest(f, newdata = tmp_data,times=c(1800), conf.int=.95)
sur5.1 = x1$surv
tmp_data$treatment = -1
x2 = survest(f, newdata = tmp_data,times=c(1800), conf.int=.95)
sur5.2 = x2$surv
test_data$delta5 = sur5.1-sur5.2

#external test cohort
x = survest(f, newdata = match_ext_data_EBV,times=c(1080), conf.int=.95)
match_ext_data_EBV$dfs3 = x$surv
x = survest(f, newdata = match_ext_data_EBV,times=c(1800), conf.int=.95)
match_ext_data_EBV$dfs5 = x$surv
tmp_data = match_ext_data_EBV[,c('EBV_4k','age','Prog_score', 'Pred_score','nomo_sig')]
tmp_data$treatment = 1
x1 = survest(f, newdata = tmp_data,times=c(1800), conf.int=.95)
sur5.1 = x1$surv
tmp_data$treatment = -1
x2 = survest(f, newdata = tmp_data,times=c(1800), conf.int=.95)
sur5.2 = x2$surv
match_ext_data_EBV$delta5 = sur5.1-sur5.2

# Kaplan-Meier curves of disease-free survival according to dichotomized 5-year survival benefit in all cohorts (Figure 5)
cl_index = c(0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
             0.7601569, 0.7601569, 0.04784314)
cl_index = matrix(cl_index, nrow = 2, ncol = 3, byrow = T)
# dev.new()
opar = par(no.readonly = T)
par(mfrow = c(3,2))
x1 = c(0,0.5,0.62,1.0)
par(fig = x1)
sub_data = subset(tra_data, delta5 > 0.0)
dd<-data.frame("surv.time" = sub_data$DFS.time, "surv.event" = sub_data$DFS,"strat" = sub_data$treatment)
km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
              x.label="",y.label="Probability of survival",main="",.col = rgb(cl_index, max = 1),
              leg.text=paste(c("CCRT", "ICT+CCRT")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
              , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))
x1 = x1+ c(0.45,0.45,0,0)
par(fig = x1, new = T)
sub_data = subset(tra_data, delta5 <= 0.0)
dd<-data.frame("surv.time" = sub_data$DFS.time, "surv.event" = sub_data$DFS,"strat" = sub_data$treatment)
km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
              x.label="",y.label="",main="",.col = rgb(cl_index, max = 1),
              leg.text=paste(c("CCRT", "ICT+CCRT")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
              , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))

x1 = c(0,0.5,0.62,1.0) - c(0,0,0.3,0.3)
par(fig = x1, new = T)
sub_data = subset(test_data, delta5 > 0.0)
dd<-data.frame("surv.time" = sub_data$DFS.time, "surv.event" = sub_data$DFS,"strat" = sub_data$treatment)
km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
              x.label="",y.label="Probability of survival",main="",.col = rgb(cl_index, max = 1),
              leg.text=paste(c("CCRT", "ICT+CCRT")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
              , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))
x1 = x1+ c(0.45,0.45,0,0)
par(fig = x1, new = T)
sub_data = subset(test_data, delta5 <= 0.0)
dd<-data.frame("surv.time" = sub_data$DFS.time, "surv.event" = sub_data$DFS,"strat" = sub_data$treatment)
km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
              x.label="",y.label="",main="",.col = rgb(cl_index, max = 1),
              leg.text=paste(c("CCRT", "ICT+CCRT")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
              , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))


x1 = c(0,0.5,0.62,1.0) - c(0,0,0.6,0.6)
par(fig = x1, new = T)
sub_data = subset(match_ext_data_EBV, delta5 > 0.0)
dd<-data.frame("surv.time" = sub_data$DFS.time, "surv.event" = sub_data$DFS,"strat" = sub_data$treatment)
km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
              x.label="",y.label="Probability of survival",main="",.col = rgb(cl_index, max = 1),
              leg.text=paste(c("CCRT", "ICT+CCRT")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
              , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))
x1 = x1+ c(0.45,0.45,0,0)
par(fig = x1, new = T)
sub_data = subset(match_ext_data_EBV, delta5 <= 0.0)
dd<-data.frame("surv.time" = sub_data$DFS.time, "surv.event" = sub_data$DFS,"strat" = sub_data$treatment)
km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
              x.label="",y.label="",main="",.col = rgb(cl_index, max = 1),
              leg.text=paste(c("CCRT", "ICT+CCRT")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
              , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))

par(opar)
