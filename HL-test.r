HLtest<-function(cal){
  len<-cal[,'n']
  meanp<-cal[,'mean.predicted']
  sump<-meanp*len
  sumy<-cal[,'KM']*len
  contr<-((sumy-sump)^2)/(len*meanp*(1-meanp))        #contribution per group to chi square
  chisqr<-sum(contr[])                                    #chi square total
  pval<-1-pchisq(chisqr,2)
  return(pval)
}