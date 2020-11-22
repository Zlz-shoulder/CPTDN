get.test.method <- function(table0){
  N<-sum(table0)
  C<-apply(table0,2,sum)
  R<-apply(table0,1,sum)
  CR<-C%*%t(R)
  M<-min(CR)
  if (N<40 | M/N<1){
    print("Fisher exect test")
    P<-fisher.test(table0)$p.value
  }else if (M/N<5){
    print("corrected Chi-square test")
    P<-chisq.test(table0)$p.value
  }else if (M/N>=5){
    print("Chi-square test")
    P<-chisq.test(table0,correct=F)$p.value
  }
  if ((P<0.07 & P>0.03)|P>0.99){
    print("Fisher exect test")
    P<-fisher.test(table0)$p.value
  }
  return(P)
}