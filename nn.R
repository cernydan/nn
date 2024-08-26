library(Rcpp)
setwd("C:/Users/danek/Desktop/mlpR/vsnn")

Rcpp::sourceCpp("rcppstuff.cpp")

dta = read.table(file="C:/Users/danek/Desktop/mlpR/podnelirano/QinQout_obs.dat")
Qkal = 100*dta$V3[1:250]/20*2
Qval = dta$V3[251:500]*100/20*2

LAG = 3

dt = matrix(0, nrow = (length(Qkal)-LAG), ncol = LAG )
for (i in 1:LAG){ dt[,i] = Qkal[(LAG-i+1):(length(Qkal)-i)] }
chtenejout = Qkal[(LAG+1):length(Qkal)]

dt2 = matrix(0, nrow = (length(Qval)-LAG), ncol = LAG )
for (i in 1:LAG){ dt2[,i] = Qval[(LAG-i+1):(length(Qval)-i)] }

mlp <- udelej_nn()
nn_set_rozmery(mlp,c(4,4,2,1))
nn_set_chtenejout(mlp,Qkal)
nn_set_traindata(mlp,dt)
nn_print_data(mlp)
nn_init_nn(mlp,LAG)
nn_online_bp_adam(mlp,1000)
simulout <- nn_get_vystupy(mlp)
nn_set_valdata(mlp,dt2)
nn_valid(mlp)
simulout2 <- nn_get_vystupy(mlp)

plot(c(Qkal,Qval),type = "l")
lines(c(simulout,simulout2),col = "red")
error <- nn_count_cost(mlp)
error

#############################################################################################xx
library(Rcpp)
setwd("C:/Users/danek/Desktop/mlpR/vsnn")

Rcpp::sourceCpp("rcppstuff.cpp")

cisloslozka <- "03"   ## číslo složky 01 až 18

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


Qkal = Qcamel$Q[1:6000]
Qval = Qcamel$Q[6001:12000]

LAG = 5

dt = matrix(0, nrow = (length(Qkal)-LAG), ncol = LAG )
for (i in 1:LAG){ dt[,i] = Qkal[(LAG-i+1):(length(Qkal)-i)] }
chtenejout = Qkal[(LAG+1):length(Qkal)]

dt2 = matrix(0, nrow = (length(Qval)-LAG), ncol = LAG )
for (i in 1:LAG){ dt2[,i] = Qval[(LAG-i+1):(length(Qval)-i)] }
chtenejout2 = Qval[(LAG+1):length(Qval)]

mlp <- udelej_nn()
nn_set_rozmery(mlp,c(10,10,10,10,1))
nn_set_chtenejout(mlp,chtenejout)
nn_set_traindata(mlp,dt)
nn_print_data(mlp)
nn_init_nn(mlp,LAG)
nn_online_bp_adam(mlp,1000)
simulout <- nn_get_vystupy(mlp)
nn_set_valdata(mlp,dt2)
nn_valid(mlp)
simulout2 <- nn_get_vystupy(mlp)
nn_set_chtenejout(mlp,chtenejout)

plot(c(Qkal,Qval),type = "l")
lines(c(simulout,simulout2),col = "red")
error <- nn_count_cost(mlp)
error


#############################################################################################xx

library(Rcpp)
setwd("C:/Users/danek/Desktop/mlpR/vsnn")

Rcpp::sourceCpp("rcppstuff.cpp")

cisloslozka <- "03"   ## číslo složky 01 až 18

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


Qcamel$Q <- (Qcamel$Q-min(Qcamel$Q))/(max(Qcamel$Q)-min(Qcamel$Q))

Qkal = Qcamel$Q[1:6000]
Qval = Qcamel$Q[6001:12000]

chtenejout = Qkal[4:6000]

mlp <- udelej_nn()
nn_set_rozmery(mlp,c(10,10,5,5,1))
nn_set_chtenejout(mlp,chtenejout)
nn_set_vstup_rada(mlp,Qkal)
nn_udelej_radky(mlp,365)
nn_init_nn(mlp,5)

nn_cnn_pokus(mlp,5)
simulout <- nn_get_vystupy(mlp)


plot(Qkal,type = "l")
lines(simulout,col = "red")
error <- nn_count_cost(mlp)
error



