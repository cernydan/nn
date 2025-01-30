{library(Rcpp)
setwd("C:/Users/danek/Desktop/mlpR/vsnn")

Rcpp::sourceCpp("rcppstuff.cpp")

dta = read.table(file="C:/Users/danek/Desktop/mlpR/podnelirano/QinQout_obs.dat")
Qkal = dta$V3[1:250]
Qval = dta$V3[251:500]

LAG = 3

dt = matrix(0, nrow = (length(Qkal)-LAG), ncol = LAG )
for (i in 1:LAG){ dt[,i] = Qkal[(LAG-i+1):(length(Qkal)-i)] }
chtenejout = Qkal[(LAG+1):length(Qkal)]

dt2 = matrix(0, nrow = (length(Qval)-LAG), ncol = LAG )
for (i in 1:LAG){ dt2[,i] = Qval[(LAG-i+1):(length(Qval)-i)] }

mlp <- udelej_nn()
nn_init_nn(mlp,LAG,c(4,4,2,1))
nn_set_chtenejout(mlp,Qkal)
nn_set_traindata(mlp,dt)
nn_shuffle_train(mlp)
nn_print_data(mlp)
nn_online_bp_adam(mlp,1000)
simulout <- nn_get_vystupy(mlp)
nn_set_valdata(mlp,dt2)
nn_valid(mlp)
simulout2 <- nn_get_vystupy(mlp)

plot(c(Qkal,Qval),type = "l")
lines(c(simulout,simulout2),col = "red")
error <- nn_count_cost(mlp)
error}
#############################################################################################xx
library(Rcpp)
setwd("C:/Users/danek/Desktop/mlpR/vsnn")

Rcpp::sourceCpp("rcppstuff.cpp")

{cisloslozka <- "18"   ## číslo složky 01 až 18

umisteni <- paste0("D:/testcamel/camel/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/usgs_streamflow/", cisloslozka)
soubory <- list.files(umisteni, pattern = "_streamflow_qc.txt$", full.names = FALSE)
cislasoubory <- data.frame(id = sub("_.*", "", soubory))
cislasoubory

poradisouboru <- 31

Qcamel <- read.table(paste0("D:/testcamel/camel/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/usgs_streamflow/",
                            cisloslozka, "/", as.character(cislasoubory$id[poradisouboru]),
                            "_streamflow_qc.txt"), header=FALSE)
VALScamel <- read.table(paste0("D:/testcamel/camel/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/nldas/",
                               cisloslozka, "/", as.character(cislasoubory$id[poradisouboru]),
                               "_lump_nldas_forcing_leap.txt"), header=FALSE, skip = 4)
names(Qcamel) = c("ID","rok","mesic","den","Q","podm")
##names(VALScamel) = c("rok", "mesic", "den", "delka dne asi[s]", "srazky [mm/den]", "")
Qcamel <- Qcamel[,1:5]
VALScamel <- VALScamel[1:length(Qcamel$Q),]
Qcamel <- Qcamel[1:length(VALScamel$V1),]
Qcamel$Q <- Qcamel$Q * 0.0283168466
Rcamel <- VALScamel$V6

Q <- (Qcamel$Q - min(Qcamel$Q))/(max(Qcamel$Q)-min(Qcamel$Q))

Qkal = Q[1:3000]
Qval = Q[3001:12000]
}

LAG = 5
pn = 50
{
dt = matrix(0, nrow = (length(Qkal)-LAG), ncol = LAG )
for (i in 1:LAG){ dt[,i] = Qkal[(LAG-i+1):(length(Qkal)-i)] }
chtenejout = Qkal[(LAG+1):length(Qkal)]

dt2 = matrix(0, nrow = (length(Qval)-LAG), ncol = LAG )
for (i in 1:LAG){ dt2[,i] = Qval[(LAG-i+1):(length(Qval)-i)] }
chtenejout2 = Qval[(LAG+1):length(Qval)]

mlp <- udelej_nn()
nn_set_chtenejout(mlp,chtenejout)
nn_set_traindata(mlp,dt)
#nn_shuffle_train(mlp)
#nn_print_data(mlp)
nn_init_nn(mlp,LAG,c(pn,pn,1))
nn_online_bp_adam(mlp,10)
simulout <- nn_get_vystupy(mlp)
nn_set_valdata(mlp,dt2)
nn_valid(mlp)
simulout2 <- nn_get_vystupy(mlp)
nn_set_chtenejout(mlp,chtenejout)

plot(c(Qkal,Qval),type = "l")
lines(c(simulout,simulout2),col = "red")
}
plot(Qval[1:300],type = "l")
lines(simulout2[1:300],col = "red")

#########################################################################################x
#########################       CNN ON FLY          #####################################
#########################################################################################

{library(Rcpp)
setwd("C:/Users/danek/Desktop/mlpR/vsnn")

Rcpp::sourceCpp("rcppstuff.cpp")
povodi <- list()
for(i in 1:5){
  
  Qcamel <- read.table(paste0("./vyberdata/",i,"_q.txt"), header=FALSE)
  Qcamel <- Qcamel[!(Qcamel$V3 == 2 & Qcamel$V4 == 29), ]  
  Rcamel <- read.table(paste0("./vyberdata/",i,"_r.txt"), header=FALSE, skip = 4)
  Rcamel <- Rcamel[!(Rcamel$V2 == 2 & Rcamel$V3 == 29), ]  
  cur_pov <- data.frame(Q = (Qcamel$V5 * 0.0283168466), R = Rcamel$V6, Tmax = Rcamel$V9)
  cur_pov$Q <- (cur_pov$Q - min(cur_pov$Q))/(max(cur_pov$Q)-min(cur_pov$Q))
  cur_pov$R <- (cur_pov$R - min(cur_pov$R))/(max(cur_pov$R)-min(cur_pov$R))
  cur_pov$Tmax <- (cur_pov$Tmax - min(cur_pov$Tmax))/(max(cur_pov$Tmax)-min(cur_pov$Tmax))
  povodi[[i]] <- cur_pov
  rm(cur_pov,Qcamel,Rcamel)
}

names(povodi) <- c("salmon","bigsur","merced","arroyo","andreas")

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

}  # nacteni Rcpp, dat a funkce kriterii

output_folder <- "D:/pokusydip/"
x = 1
vysl = data.frame()
for(n in 1){
for(m in 1){
for(l in 1){
for(k in 1){
for(j in 1){
for(i in 5){
  row_ker = 3
  col_ker = 4
  poc_ker = 100
  roky_cal = 20
  iter = 500
  
  mlp <- udelej_nn()
  nn_init_nn(mlp,poc_ker,c(poc_ker,poc_ker,1))
  nn_set_vstup_rady(mlp, povodi[[i]]$Q[1:(roky_cal*365)],
                    povodi[[i]]$Q[(roky_cal*365+1):length(povodi[[i]]$Q)], 
                    povodi[[i]]$R[1:(roky_cal*365)],
                    povodi[[i]]$R[(roky_cal*365+1):length(povodi[[i]]$R)], 
                    povodi[[i]]$Tmax[1:(roky_cal*365)],
                    povodi[[i]]$Tmax[(roky_cal*365+1):length(povodi[[i]]$Tmax)]
  )
  nn_set_chtenejout(mlp,povodi[[i]]$Q[((row_ker-1)*365+col_ker+1):(365*roky_cal+col_ker)])
  nn_cnn_onfly_cal(mlp,row_ker,col_ker,poc_ker,iter)
  nn_cnn_onfly_val(mlp)
  vystupy <- nn_get_vystupy(mlp)
  
  file_path <- file.path(output_folder, paste0(x, ".png"))
  png(file_path, width = 800, height = 600) # Nastavení výstupního souboru
  plot.new()
  plot(
    c(povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1+col_ker):length(povodi[[i]]$Q)],
      povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1):((roky_cal+row_ker-1)*365+col_ker)]),
    type = "l",
    col = "black",
    main = paste0("pokus=", j," povodi=",i," row_ker=", row_ker," col_ker=", col_ker, " poc=", poc_ker, " roky cal=", roky_cal, " iter=", iter),
    sub = paste0(" mae=", mae(vystupy,c(povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1+col_ker):length(povodi[[i]]$Q)],
                                        povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1):((roky_cal+row_ker-1)*365+col_ker)])),
                 " rmse=", rmse(vystupy,c(povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1+col_ker):length(povodi[[i]]$Q)],
                                          povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1):((roky_cal+row_ker-1)*365+col_ker)])),
                 " nse=", nse(vystupy,c(povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1+col_ker):length(povodi[[i]]$Q)],
                                        povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1):((roky_cal+row_ker-1)*365+col_ker)])))
    
  )
  lines(vystupy,col = "red")
  dev.off() # Ukončení záznamu do souboru
  print(x)
  vysl[x,1] = j
  vysl[x,2] = i
  vysl[x,3] = row_ker
  vysl[x,4] = col_ker
  vysl[x,5] = poc_ker
  vysl[x,6] = roky_cal
  vysl[x,7] = iter
  vysl[x,8] = mae(vystupy,c(povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1+col_ker):length(povodi[[i]]$Q)],
                            povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1):((roky_cal+row_ker-1)*365+col_ker)]))
  vysl[x,9] = rmse(vystupy,c(povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1+col_ker):length(povodi[[i]]$Q)],
                             povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1):((roky_cal+row_ker-1)*365+col_ker)]))
  vysl[x,10] = nse(vystupy,c(povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1+col_ker):length(povodi[[i]]$Q)],
                            povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1):((roky_cal+row_ker-1)*365+col_ker)]))
  x= x+1
}}}}}}
names(vysl) = c("pokus","povodi","row_ker","col_ker","poc","roky_cal","iter","mae","rmse","nse")
vysl[vysl$nse == max(vysl$nse),]
vysl[vysl$rmse == min(vysl$rmse),]
vysl[vysl$mae == min(vysl$mae),]
vysl[vysl$nse>0.5&vysl$povodi==2,]

#########################################################################################x
#########################       CNN ON FLY          #####################################
#########################################################################################

{library(Rcpp)
setwd("C:/Users/danek/Desktop/mlpR/vsnn")

Rcpp::sourceCpp("rcppstuff.cpp")
povodi <- list()
for(i in 1:5){
  
  Qcamel <- read.table(paste0("./vyberdata/",i,"_q.txt"), header=FALSE)
  Qcamel <- Qcamel[!(Qcamel$V3 == 2 & Qcamel$V4 == 29), ]  
  Rcamel <- read.table(paste0("./vyberdata/",i,"_r.txt"), header=FALSE, skip = 4)
  Rcamel <- Rcamel[!(Rcamel$V2 == 2 & Rcamel$V3 == 29), ]  
  cur_pov <- data.frame(Q = (Qcamel$V5 * 0.0283168466), R = Rcamel$V6, Tmax = Rcamel$V9)
  cur_pov$Q <- (cur_pov$Q - min(cur_pov$Q))/(max(cur_pov$Q)-min(cur_pov$Q))
  cur_pov$R <- (cur_pov$R - min(cur_pov$R))/(max(cur_pov$R)-min(cur_pov$R))
  cur_pov$Tmax <- (cur_pov$Tmax - min(cur_pov$Tmax))/(max(cur_pov$Tmax)-min(cur_pov$Tmax))
  povodi[[i]] <- cur_pov
  rm(cur_pov,Qcamel,Rcamel)
}

names(povodi) <- c("salmon","bigsur","merced","dry","palm")

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

}  # nacteni Rcpp, dat a funkce kriterii

output_folder <- "D:/pokusydip/"
x = 1
vysl = data.frame()
for(n in 1){
for(m in 1){
for(l in 1){
for(k in 1){
for(j in 1){
for(i in 4){
  row_ker = 1
  col_ker = 7
  poc_ker = 50
  roky_cal = 15
  iter = 1000
  
  mlp <- udelej_nn()
  nn_init_nn(mlp,poc_ker,c(poc_ker,poc_ker,1))
  nn_set_vstup_rady(mlp, povodi[[i]]$Q[1:(roky_cal*365)],
                    povodi[[i]]$Q[(roky_cal*365+1):length(povodi[[i]]$Q)], 
                    povodi[[i]]$R[1:(roky_cal*365)],
                    povodi[[i]]$R[(roky_cal*365+1):length(povodi[[i]]$R)], 
                    povodi[[i]]$Tmax[1:(roky_cal*365)],
                    povodi[[i]]$Tmax[(roky_cal*365+1):length(povodi[[i]]$Tmax)]
  )
  nn_set_chtenejout(mlp,povodi[[i]]$Q[((row_ker-1)*365+col_ker+1):(365*roky_cal+col_ker)])
  nn_cnn_onfly_cal(mlp,row_ker,col_ker,poc_ker,iter)
  nn_cnn_onfly_val(mlp)
  vystupy <- nn_get_vystupy(mlp)
  
  file_path <- file.path(output_folder, paste0(x, ".png"))
  png(file_path, width = 800, height = 600) # Nastavení výstupního souboru
  plot.new()
  plot(
    c(povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1+col_ker):length(povodi[[i]]$Q)],
      povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1):((roky_cal+row_ker-1)*365+col_ker)]),
    type = "l",
    col = "black",
    main = paste0("pokus=", j," povodi=",i," row_ker=", row_ker," col_ker=", col_ker, " poc=", poc_ker, " roky cal=", roky_cal, " iter=", iter),
    sub = paste0(" mae=", mae(vystupy,c(povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1+col_ker):length(povodi[[i]]$Q)],
                                        povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1):((roky_cal+row_ker-1)*365+col_ker)])),
                 " rmse=", rmse(vystupy,c(povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1+col_ker):length(povodi[[i]]$Q)],
                                          povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1):((roky_cal+row_ker-1)*365+col_ker)])),
                 " nse=", nse(vystupy,c(povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1+col_ker):length(povodi[[i]]$Q)],
                                        povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1):((roky_cal+row_ker-1)*365+col_ker)])))
    
  )
  lines(vystupy,col = "red")
  dev.off() # Ukončení záznamu do souboru
  print(x)
  vysl[x,1] = j
  vysl[x,2] = i
  vysl[x,3] = row_ker
  vysl[x,4] = col_ker
  vysl[x,5] = poc_ker
  vysl[x,6] = roky_cal
  vysl[x,7] = iter
  vysl[x,8] = mae(vystupy,c(povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1+col_ker):length(povodi[[i]]$Q)],
                            povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1):((roky_cal+row_ker-1)*365+col_ker)]))
  vysl[x,9] = rmse(vystupy,c(povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1+col_ker):length(povodi[[i]]$Q)],
                             povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1):((roky_cal+row_ker-1)*365+col_ker)]))
  vysl[x,10] = nse(vystupy,c(povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1+col_ker):length(povodi[[i]]$Q)],
                            povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1):((roky_cal+row_ker-1)*365+col_ker)]))
  x= x+1
}}}}}}
names(vysl) = c("pokus","povodi","row_ker","col_ker","poc","roky_cal","iter","mae","rmse","nse")
vysl[vysl$nse == max(vysl$nse),]
vysl[vysl$rmse == min(vysl$rmse),]
vysl[vysl$mae == min(vysl$mae),]
vysl[vysl$nse>0.5&vysl$povodi==2,]


#########################################################################################x
#########################         1D CNN          #####################################
#########################################################################################

{library(Rcpp)
setwd("C:/Users/danek/Desktop/mlpR/vsnn")

Rcpp::sourceCpp("rcppstuff.cpp")
povodi <- list()
for(i in 1:5){
  
  Qcamel <- read.table(paste0("./vyberdata/",i,"_q.txt"), header=FALSE)
  Qcamel <- Qcamel[!(Qcamel$V3 == 2 & Qcamel$V4 == 29), ]  
  Rcamel <- read.table(paste0("./vyberdata/",i,"_r.txt"), header=FALSE, skip = 4)
  Rcamel <- Rcamel[!(Rcamel$V2 == 2 & Rcamel$V3 == 29), ]  
  cur_pov <- data.frame(Q = (Qcamel$V5 * 0.0283168466), R = Rcamel$V6, Tmax = Rcamel$V9)
  cur_pov$Q <- (cur_pov$Q - min(cur_pov$Q))/(max(cur_pov$Q)-min(cur_pov$Q))
  cur_pov$R <- (cur_pov$R - min(cur_pov$R))/(max(cur_pov$R)-min(cur_pov$R))
  cur_pov$Tmax <- (cur_pov$Tmax - min(cur_pov$Tmax))/(max(cur_pov$Tmax)-min(cur_pov$Tmax))
  povodi[[i]] <- cur_pov
  rm(cur_pov,Qcamel,Rcamel)
}

names(povodi) <- c("salmon","bigsur","merced","dry","palm")

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

}  # nacteni Rcpp, dat a funkce kriterii

output_folder <- "D:/pokusydip/"
x = 1
vysl = data.frame()
for(o in 1){
for(n in 2){
for(m in 1){
for(l in 1){
for(k in 1){
for(j in 1){
for(i in 3){
  ker = 2+n
  poc_ker = 10*m
  roky_cal = 10+5*l
  iter = 25*k
  velic = o
  
  mlp <- udelej_nn()
  nn_init_nn(mlp,poc_ker,c(poc_ker,poc_ker,1))
  nn_set_vstup_rady(mlp, povodi[[i]]$Q[1:(roky_cal*365)],
                    povodi[[i]]$Q[(roky_cal*365+1):length(povodi[[i]]$Q)], 
                    povodi[[i]]$R[1:(roky_cal*365)],
                    povodi[[i]]$R[(roky_cal*365+1):length(povodi[[i]]$R)], 
                    povodi[[i]]$Tmax[1:(roky_cal*365)],
                    povodi[[i]]$Tmax[(roky_cal*365+1):length(povodi[[i]]$Tmax)]
  )
  nn_set_chtenejout(mlp,povodi[[i]]$Q[(ker+1):(365*roky_cal+1)])
  nn_cnn_1d_cal(mlp,ker,poc_ker,iter,velic)
  nn_cnn_1d_val(mlp,velic)
  vystupy <- nn_get_vystupy(mlp)
  
  file_path <- file.path(output_folder, paste0(x, ".png"))
  png(file_path, width = 800, height = 600) # Nastavení výstupního souboru
  plot.new()
  plot(
    povodi[[i]]$Q[(roky_cal*365+1+ker):length(povodi[[i]]$Q)],
    type = "l",
    col = "black",
    main = paste0("pokus=", j," povodi=",i," velic=",velic," ker=",ker, " poc=", poc_ker, " roky cal=", roky_cal, " iter=", iter),
    sub = paste0(" mae=", mae(vystupy,povodi[[i]]$Q[(roky_cal*365+1+ker):length(povodi[[i]]$Q)]),
                 " rmse=", rmse(vystupy,povodi[[i]]$Q[(roky_cal*365+1+ker):length(povodi[[i]]$Q)]),
                 " nse=", nse(vystupy,povodi[[i]]$Q[(roky_cal*365+1+ker):length(povodi[[i]]$Q)]))
    
  )
  lines(vystupy,col = "red")
  dev.off() # Ukončení záznamu do souboru
  print(x)
  vysl[x,1] = j
  vysl[x,2] = i
  vysl[x,3] = velic
  vysl[x,4] = ker
  vysl[x,5] = poc_ker
  vysl[x,6] = roky_cal
  vysl[x,7] = iter
  vysl[x,8] = mae(vystupy,povodi[[i]]$Q[(roky_cal*365+1+ker):length(povodi[[i]]$Q)])
  vysl[x,9] = rmse(vystupy,povodi[[i]]$Q[(roky_cal*365+1+ker):length(povodi[[i]]$Q)])
  vysl[x,10] = nse(vystupy,povodi[[i]]$Q[(roky_cal*365+1+ker):length(povodi[[i]]$Q)])
  x= x+1
}}}}}}}
names(vysl) = c("pokus","povodi","velic","ker","poc","roky_cal","iter","mae","rmse","nse")
vysl[vysl$nse == max(vysl$nse),]
vysl[vysl$rmse == min(vysl$rmse),]
vysl[vysl$mae == min(vysl$mae),]
vysl[vysl$nse>0.5&vysl$povodi==2,]

#########################################################################################x
#########################         FULL CNN          #####################################
#########################################################################################

{library(Rcpp)
setwd("C:/Users/danek/Desktop/mlpR/vsnn")

Rcpp::sourceCpp("rcppstuff.cpp")
povodi <- list()
for(i in 1:5){
  
  Qcamel <- read.table(paste0("./vyberdata/",i,"_q.txt"), header=FALSE)
  Qcamel <- Qcamel[!(Qcamel$V3 == 2 & Qcamel$V4 == 29), ]  
  Rcamel <- read.table(paste0("./vyberdata/",i,"_r.txt"), header=FALSE, skip = 4)
  Rcamel <- Rcamel[!(Rcamel$V2 == 2 & Rcamel$V3 == 29), ]  
  cur_pov <- data.frame(Q = (Qcamel$V5 * 0.0283168466), R = Rcamel$V6, Tmax = Rcamel$V9)
  cur_pov$Q <- (cur_pov$Q - min(cur_pov$Q))/(max(cur_pov$Q)-min(cur_pov$Q))
  cur_pov$R <- (cur_pov$R - min(cur_pov$R))/(max(cur_pov$R)-min(cur_pov$R))
  cur_pov$Tmax <- (cur_pov$Tmax - min(cur_pov$Tmax))/(max(cur_pov$Tmax)-min(cur_pov$Tmax))
  povodi[[i]] <- cur_pov
  rm(cur_pov,Qcamel,Rcamel)
}

names(povodi) <- c("salmon","bigsur","merced","dry","palm")

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

}  # nacteni Rcpp, dat a funkce kriterii

output_folder <- "D:/pokusydip/"
x = 1
vysl = data.frame()

for(j in 1){
for(i in 3){
  iter = 1
  
  mlp <- udelej_nn()
  nn_init_nn(mlp,50,c(50,50,1))
  nn_set_vstup_rady(mlp, povodi[[i]]$Q[1:600],
                    povodi[[i]]$Q[601:12000], 
                    povodi[[i]]$R[1:600],
                    povodi[[i]]$R[601:12000], 
                    povodi[[i]]$Tmax[1:600],
                    povodi[[i]]$Tmax[601:12000]
  )
  nn_cnn_full_cal(mlp,iter)
  nn_cnn_full_val(mlp)
  vystupy <- nn_get_vystupy(mlp)
  
  file_path <- file.path(output_folder, paste0(x, ".png"))
  png(file_path, width = 800, height = 600) # Nastavení výstupního souboru
  plot.new()
  plot(
    povodi[[i]]$Q[povodi[[i]]$Q[1001:12020]],
    type = "l",
    col = "black",
    main = paste0("pokus=", j," povodi=",i," iter=", iter),
    sub = paste0(" mae=", mae(vystupy,povodi[[i]]$Q[1001:12020]),
                 " rmse=", rmse(vystupy,povodi[[i]]$Q[1001:12020]),
                 " nse=", nse(vystupy,povodi[[i]]$Q[1001:12020]))
    
  )
  lines(vystupy,col = "red")
  dev.off() # Ukončení záznamu do souboru
  print(x)
  vysl[x,1] = j
  vysl[x,2] = i
  vysl[x,3] = iter
  vysl[x,4] = mae(vystupy,povodi[[i]]$Q[1001:12020])
  vysl[x,5] = rmse(vystupy,povodi[[i]]$Q[1001:12020])
  vysl[x,6] = nse(vystupy,povodi[[i]]$Q[1001:12020])
  x= x+1
}}
names(vysl) = c("pokus","povodi","iter","mae","rmse","nse")
vysl[vysl$nse == max(vysl$nse),]
vysl[vysl$rmse == min(vysl$rmse),]
vysl[vysl$mae == min(vysl$mae),]
vysl[is.na(vysl$nse)==FALSE&vysl$nse>0.952,]

