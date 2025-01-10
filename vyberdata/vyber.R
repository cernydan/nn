setwd("C:/Users/danek/Desktop/mlpR/vsnn")

povodi <- list()
for(i in 1:5){
  
  Qcamel <- read.table(paste0("./vyberdata/",i,"_q.txt"), header=FALSE)
  Rcamel <- read.table(paste0("./vyberdata/",i,"_r.txt"), header=FALSE, skip = 4)
  cur_pov <- data.frame(Q = (Qcamel$V5 * 0.0283168466), R = Rcamel$V6, Tmax = Rcamel$V9)
  cur_pov$Q <- (cur_pov$Q - min(cur_pov$Q))/(max(cur_pov$Q)-min(cur_pov$Q))
  cur_pov$R <- (cur_pov$R - min(cur_pov$R))/(max(cur_pov$R)-min(cur_pov$R))
  cur_pov$Tmax <- (cur_pov$Tmax - min(cur_pov$Tmax))/(max(cur_pov$Tmax)-min(cur_pov$Tmax))
  povodi[[i]] <- cur_pov
  rm(cur_pov,Qcamel,Rcamel)
}

names(povodi) <- c("salmon","bigsur","merced","naci","sft")
povodi[["salmon"]]

plot(povodi[["merced"]]$Q,type = "l")

mae <- function(mod, obs) {
  if (length(mod) != length(obs)) {
    stop("Vektory musí mít stejnou délku")
  }
  err = 0
  for (i in 1:length(mod)){
    err = err + abs(mod[i] - obs[i])
  }
  err = err/length(mod)
  return(err)
}

abc = c(1,4,8,5)
def = c(3,1,8,9)


rmse <- function(mod, obs) {
  if (length(mod) != length(obs)) {
    stop("Vektory musí mít stejnou délku")
  }
  err = 0
  for (i in 1:length(mod)){
    err = err + (mod[i] - obs[i])^2
  }
  err = sqrt(err/length(mod))
  return(err)
}

nse <- function(mod, obs) {
  if (length(mod) != length(obs)) {
    stop("Vektory musí mít stejnou délku")
  }
  err = 0
  cit = 0
  jme = 0
  for (i in 1:length(mod)){
    cit = cit + (mod[i] - obs[i])^2
    jme = jme + (mod[i] - mean(obs))^2
  }
  err = 1-(cit/length(mod))/(jme/length(mod))
  return(err)
}

