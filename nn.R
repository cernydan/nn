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

#####################################################################################################x

library(Rcpp)
setwd("C:/Users/danek/Desktop/mlpR/vsnn")
Rcpp::sourceCpp("rcppstuff.cpp")

dta = read.table(file="C:/Users/danek/Desktop/mlpR/podnelirano/QinQout_obs.dat")
Qkal = dta$V3[1:250]
Qval = dta$V3[251:500]

LAG = 4
{
  dt = matrix(0, nrow = (length(Qkal)-LAG), ncol = LAG )
  for (i in 1:LAG){ dt[,i] = Qkal[(LAG-i+1):(length(Qkal)-i)] }
  chtenejout = Qkal[(LAG+1):length(Qkal)]

  
  mlp <- udelej_nn()
  nn_set_chtenejout(mlp,chtenejout)
  nn_set_traindata(mlp,dt)
  #nn_shuffle_train(mlp)
  #nn_print_data(mlp)
  lstm_1cell(mlp,6,10000)
  simulout <- nn_get_vystupy(mlp)

  
  plot(c(Qkal),type = "l")
  lines(c(simulout),col = "red")
  error <- nn_count_cost(mlp)
  error
}

##########################################x    CNN POKUS
library(Rcpp)
setwd("C:/Users/danek/Desktop/mlpR/vsnn")
Rcpp::sourceCpp("rcppstuff.cpp")

{
cisloslozka <- "03"   ## číslo složky 01 až 18

umisteni <- paste0("D:/testcamel/camel/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/usgs_streamflow/", cisloslozka)
soubory <- list.files(umisteni, pattern = "_streamflow_qc.txt$", full.names = FALSE)
cislasoubory <- data.frame(id = sub("_.*", "", soubory))
cislasoubory

poradisouboru <- 11

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

Qcamel <- Qcamel[!(Qcamel$mesic == 2 & Qcamel$den == 29), ]   

Q <- (Qcamel$Q - min(Qcamel$Q))/(max(Qcamel$Q)-min(Qcamel$Q))
}

ker = 7
poc_ker = 20
roky_cal = 20
roky_val = 20

vstup_cal <- Q[1:(roky_cal*365)]
chtenejout_cal <- Q[((ker-1)*365+ker+1):(365*roky_cal+ker)]

mlp <- udelej_nn()
nn_init_nn(mlp,poc_ker,c(poc_ker,poc_ker,1))
nn_set_vstup_rada(mlp,vstup_cal)
nn_set_chtenejout(mlp,chtenejout_cal)
nn_cnn_pokus_cal(mlp,ker,poc_ker,50)
simulout_cal <- nn_get_vystupy(mlp)

plot(chtenejout_cal,type = "l")
lines(simulout_cal,col = "red")

plot(chtenejout_cal[550:850],type = "l",ylim = c(0.1,0.3))
lines(simulout_cal[550:850],col = "red")

vstup_val <- Q[3651:10950]
chtenejout_val <- Q[5848:10957]

nn_set_vstup_rada(mlp,vstup_val)
nn_set_chtenejout(mlp,chtenejout_val)
nn_cnn_pokus_val(mlp)
simulout_val <- nn_get_vystupy(mlp)

plot(chtenejout_val,type = "l")
lines(simulout_val,col = "red")


x = 1
vystupy = list()

for(i in 1:3){
  for(j in 1:4){
    for(k in 1:3){
      print(x)
      
      ker = 7
      poc_ker = i*10
      roky_cal = 5+5*j
      
      
      vstup_cal <- Q[1:(roky_cal*365)]
      chtenejout_cal <- Q[((ker-1)*365+ker+1):(365*roky_cal+ker)]
      
      mlp <- udelej_nn()
      nn_init_nn(mlp,poc_ker,c(poc_ker,poc_ker,1))
      nn_set_vstup_rada(mlp,vstup_cal)
      nn_set_chtenejout(mlp,chtenejout_cal)
      nn_cnn_pokus_cal(mlp,ker,poc_ker,(50*k))
      vystupy[[x]] <- nn_get_vystupy(mlp)
      x=x+1
    }
  }
}
vystupy


# Nastavení složky, kam se obrázky uloží
output_folder <- "C:/Users/danek/Desktop/grafy"

# Iterace přes list a uložení grafů
for (i in seq_along(vystupy)) {

  
  # Vytvoření cesty pro uložení souboru
  file_path <- file.path(output_folder, paste0(i, ".png"))
  
  # Uložení grafu jako PNG

  png(file_path, width = 800, height = 600) # Nastavení výstupního souboru
  plot.new()
  plot(
    chtenejout_cal,
    type = "l",
    col = "black",

  )
  lines(vystupy[[i]],col = "red")
  dev.off() # Ukončení záznamu do souboru
}





