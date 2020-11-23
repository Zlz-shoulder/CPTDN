############################################
#Obtain matched patients with similar baseline characteristics between treatment
#using nearest neighbor matching method
############################################

pacman::p_load(knitr, wakefield, MatchIt, tableone, captioner)
#Processing data form center 1
wb = 'RawData_center1.csv'
clinic_data =  read.csv(wb, header = TRUE, sep = ',')
row.names(clinic_data) = clinic_data$Pat_ID
names(clinic_data)
#treatment=0: CCRT, treatment=1: ICT+CCRT
clinic_data$treatment = ifelse(clinic_data$treatment<0.5,0,1)
#Truncate clinical factor
clinic_data$age = clinic_data$age/100
clinic_data$EBV_DNA = ifelse(clinic_data$EBV_DNA>16000,16000,clinic_data$EBV_DNA)/16000
clinic_data$EBV_4k = ifelse(clinic_data$EBV_DNA>16000/4000,1,0)
clinic_data$LDH = ifelse(clinic_data$LDH>400,400,clinic_data$LDH)/400
clinic_data$LDHcut = ifelse(clinic_data$LDH<120/400 | clinic_data$LDH>250/400,1,0)
clinic_data$HGB = ifelse(clinic_data$HGB>200,200,clinic_data$HGB)/200
clinic_data$HGBcut = ifelse(clinic_data$sex == 0,ifelse(clinic_data$HGB<130/200 | clinic_data$HGB>175/200,1,0),ifelse(clinic_data$HGB<115/200 | clinic_data$HGB>150/200,1,0))
clinic_data$WHOcut = ifelse(clinic_data$WHO>1,1,0)

#Match baseline characteristics between treatment
set.seed(1234)
match.it <- matchit(treatment1 ~Ncut+sex + age + HGBcut + EBV_4k + drinkingcut + His_cancercut + smokingcut + tumor_volume,
                    data = clinic_data, method="nearest", discard = "both",ratio=1, caliper=0.01)
df.match <- match.data(match.it)[1:ncol(clinic_data)]


#Baseline characteristics differences between treatment after nonparametric match
#####
#Continuous variables were tested by either the Kruskal-Wallis rank sum test
tmp_data = df.match
summary(tmp_data$tumor_volume)
x1 = tmp_data$tumor_volume
group = tmp_data$treatment
kruskal.test(x1,factor(group))


summary(tmp_data$age)
x1 = tmp_data$age
group = tmp_data$treatment
kruskal.test(x1,factor(group))

#categorical variables were tested by either the Pearson’s χ2 test or the Fisher’s exact test.
source('chisq-fisher-test.r')
name =  c("sex","HGBcut","CRPcut","LDHcut","ALBcut","EBV_4k","smokingcut","drinkingcut","His_cancercut")
for(s in name){
  x1 = tmp_data[,s]
  mark = is.na(x1)
  print("No of the loss:")
  print(sum(mark))
  var = x1[!mark]
  group = tmp_data$treatment[!mark]
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

#Divide patients into different datasets
df.match$treatment = ifelse(df.match$treatment<0.5,-1,1)
nn = nrow(df.match)
set.seed(012)
x1 = sample(1:nn, 0.6*nn, replace = F)
x2 = sample(x1, 0.2*nn, replace = F)
x3 = setdiff(x=1:nn, x1)
x4 = sample(x3, 0.2*length(x3), replace = F)
ind_test = setdiff(x=x3, x4)
ind_val = c(x2,x4)
#Dataset for training deep neural networks (data_cohort=1)
df.match$data_cohort = 0
df.match[x1, 'data_cohort'] = 1
#Dataset for debuging network hyperparameters (data_cohort1=1)
df.match$data_cohort1 = 0
df.match[ind_val, 'data_cohort1'] = 1
#Training cohort (data_cohort2=0), internal test cohort (data_cohort2=1)
df.match$data_cohort2 = 0
df.match[ind_test, 'data_cohort2'] = 1
#save matched data from center 1 for analysis
wb = 'matched_data_from_center1.csv'
write.csv(df.match, file = wb, row.names = F)

##########################################################
##########################################################
#Processing data form center 2-4
wb = 'RawData_center2-4.csv'
test_extra = read.csv(wb, header = TRUE, sep = ',')
row.names(test_extra) = test_extra$Pat_ID
test_extra$treatment = ifelse(test_extra$treatment<0.5,0,1)
#Truncate clinical factor
test_extra$age = test_extra$age/100
test_extra$EBV_DNA = ifelse(test_extra$EBV_DNA>16000,16000,test_extra$EBV_DNA)/16000
test_extra$EBV_4k = ifelse(test_extra$EBV_DNA>16000/4000,1,0)
test_extra$LDH = ifelse(test_extra$LDH>400,400,test_extra$LDH)/400
test_extra$LDHcut = ifelse(test_extra$LDH<120/400 | test_extra$LDH>250/400,1,0)
test_extra$HGB = ifelse(test_extra$HGB>200,200,test_extra$HGB)/200
test_extra$HGBcut = ifelse(test_extra$sex == 0,ifelse(test_extra$HGB<130/200 | test_extra$HGB>175/200,1,0),ifelse(test_extra$HGB<115/200 | test_extra$HGB>150/200,1,0))
test_extra$WHOcut = ifelse(test_extra$WHO>1,1,0)



#missing data in categorical variables are randomly assigned according to the distribution of the recorded data.
extra_tmp_data = test_extra[,c('treatment', 'LDHcut','HGBcut','EBV_4k','sex','age','tumor_volume','drinkingcut')]
name = c('HGBcut','EBV_4k', 'LDHcut','drinkingcut')
for(s in name){
  x1 = extra_tmp_data[, s]
  mask = is.na(x1)
  nn = sum(mask)
  x2 = rep(0, nn)
  set.seed(1234)
  otn = round(nn*sum(x1[!mask])/length(x1[!mask]))
  ind = sample(1:nn, otn, replace = F)
  x2[ind] = 1
  x1[mask] = x2
  extra_tmp_data[, s] = x1
}
# Match baseline characteristics between treatment 
set.seed(1234)
match.it_ext <- matchit(treatment ~sex+ age + HGBcut + LDHcut  + EBV_4k + tumor_volume,
                        data = extra_tmp_data, method="nearest", discard = "both",ratio=1, caliper=0.1)
df.match_ext <- match.data(match.it_ext)[1:ncol(extra_tmp_data)]


#Baseline characteristics differences between treatment after nonparametric match
#####
#Continuous variables were tested by either the Kruskal-Wallis rank sum test
name = row.names(df.match_ext)
match_ext_data = test_extra[name, ]


tmp_data = match_ext_data
summary(tmp_data$tumor_volume)
x1 = tmp_data$tumor_volume
group = tmp_data$treatment
kruskal.test(x1,factor(group))


summary(tmp_data$age)
x1 = tmp_data$age
group = tmp_data$treatment
kruskal.test(x1,factor(group))

#categorical variables were tested by either the Pearson’s χ2 test or the Fisher’s exact test.
name =  c("sex","HGBcut","CRPcut","LDHcut","ALBcut","EBV_4k","smokingcut","drinkingcut","His_cancercut")
for(s in name){
  x1 = tmp_data[,s]
  mark = is.na(x1)
  print("No of the loss:")
  print(sum(mark))
  var = x1[!mark]
  group = tmp_data$treatment[!mark]
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

#save data from center 2-4 for analysis
wb = 'matched_data_from_center2-4.csv'
write.csv(test_extra, file = wb, row.names = F)
