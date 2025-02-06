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
#########################       CNN ON FLY  POKUS        #####################################
#########################################################################################

{library(Rcpp)
  setwd("C:/Users/danek/Desktop/mlpR/vsnn")
  library(ggplot2)
  Rcpp::sourceCpp("rcppstuff.cpp")
  povodi <- list()
  minmax <- data.frame()
  for(i in 1:5){
    Qcamel <- read.table(paste0("./vyberdata/",i,"_q.txt"), header=FALSE)
    Qcamel <- Qcamel[!(Qcamel$V3 == 2 & Qcamel$V4 == 29), ]  
    Rcamel <- read.table(paste0("./vyberdata/",i,"_r.txt"), header=FALSE, skip = 4)
    Rcamel <- Rcamel[!(Rcamel$V2 == 2 & Rcamel$V3 == 29), ]  
    cur_pov <- data.frame(Datum = as.Date(paste(Rcamel$V1,Rcamel$V2,Rcamel$V3, sep = "-")),
                          Q = (Qcamel$V5 * 0.0283168466), R = Rcamel$V6, Tmax = Rcamel$V9)
    minmax[i,1] = min(cur_pov$Q)
    minmax[i,2] = max(cur_pov$Q)
    cur_pov$Q <- (cur_pov$Q - min(cur_pov$Q))/(max(cur_pov$Q)-min(cur_pov$Q))
    cur_pov$R <- (cur_pov$R - min(cur_pov$R))/(max(cur_pov$R)-min(cur_pov$R))
    cur_pov$Tmax <- (cur_pov$Tmax - min(cur_pov$Tmax))/(max(cur_pov$Tmax)-min(cur_pov$Tmax))
    povodi[[i]] <- cur_pov
    rm(cur_pov,Qcamel,Rcamel)
  }
  names(minmax) <- c("minQ","maxQ")
  names_povodi <- c("salmon","bigsur","merced","arroyo","andreas")
  names(povodi) <- names_povodi
  
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
      cit = cit + (obs[i] - mod[i])^2
      jme = jme + (obs[i] - mean(obs))^2
    }
    err = 1-(cit/length(mod))/(jme/length(mod))
    return(err)
  }
  
  pi <- function(mod, obs) {
    if (length(mod) != length(obs)) {
      stop("Vektory musí mít stejnou délku")
    }
    err = 0
    cit = 0
    jme = 0
    for (i in 2:length(mod)){
      cit = cit + (obs[i] - mod[i])^2
      jme = jme + (obs[i] - obs[i-1])^2
    }
    err = 1-(cit/length(mod))/(jme/length(mod))
    return(err)
  }
  
}  # nacteni Rcpp, dat a funkce kriterii

output_folder <- "D:/pokusydip/"
x = 1
vysl = data.frame()
vysl_cr = list()
for(n in 2){
  for(m in 3){
    for(l in 30){
      for(k in 20){
        for(j in 1000){
          for(i in 5){
            row_ker = n
            col_ker = m
            poc_ker = l
            roky_cal = k
            iter = j
            
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
            
            # Vytvoření datového rámce pro ggplot
            df_plot <- data.frame(
              Datum = povodi[[i]]$Datum[((roky_cal+row_ker-1)*365+1):length(povodi[[i]]$Datum)],
              Q_měřené = c(povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1+col_ker):length(povodi[[i]]$Q)],
                           povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1):((roky_cal+row_ker-1)*365+col_ker)])*(minmax[i,2]-minmax[i,1])+minmax[i,1],
              Q_modelované = vystupy*(minmax[i,2]-minmax[i,1])+minmax[i,1]
            )
            
            # Vytvoření ggplot objektu
            plot <- ggplot(df_plot, aes(x = Datum)) +
              geom_line(aes(y = Q_měřené, color = "Měřené"), linewidth = 0.6) +
              geom_line(aes(y = Q_modelované, color = "Model"), linewidth = 0.45) +
              scale_color_manual(values = c("Měřené" = "black", "Model" = "red")) +
              labs(
                title = paste0("povodi=", i, " radky=", row_ker, " sloupce=", col_ker, 
                               " poc=", poc_ker, " roky cal=", roky_cal, " iter=", iter),
                subtitle = paste0(
                  "mae=", round(mae(df_plot$Q_modelované, df_plot$Q_měřené),3), 
                  "    rmse=", round(rmse(df_plot$Q_modelované, df_plot$Q_měřené),3), 
                  "    nse=", round(nse(df_plot$Q_modelované, df_plot$Q_měřené),3), 
                  "    pi=", round(pi(df_plot$Q_modelované, df_plot$Q_měřené),3)
                ),
                x = NULL,
                y = "Q [m3/s]",
                colour = NULL
              ) +
              theme_minimal() +
              theme(
                legend.position = "bottom",
                plot.title = element_text(size = 14, face = "bold"),
                plot.subtitle = element_text(size = 10)
              )
            
            cur_id = paste0(i,"_",row_ker,"_",col_ker,"_",poc_ker,"_",roky_cal,"_",iter,"_o")
            # Uložení grafu do souboru
            file_path <- file.path(output_folder, paste0(cur_id, ".png"))
            ggsave(file_path, plot = plot, width = 8, height = 6, dpi = 300, bg = "white")
            
            print(x)
            vysl[x,1] = names_povodi[i]
            vysl[x,2] = row_ker
            vysl[x,3] = col_ker
            vysl[x,4] = poc_ker
            vysl[x,5] = roky_cal
            vysl[x,6] = iter
            vysl[x,7] = round(mae(df_plot$Q_modelované,df_plot$Q_měřené),3)
            vysl[x,8] = round(rmse(df_plot$Q_modelované,df_plot$Q_měřené),3)
            vysl[x,9] = round(nse(df_plot$Q_modelované,df_plot$Q_měřené),3)
            vysl[x,10] = round(pi(df_plot$Q_modelované,df_plot$Q_měřené),3)
            vysl[x,11] = x
            vysl[x,12] = cur_id
            vysl_cr[[x]] <- df_plot
            x= x+1
          }}}}}}
names(vysl) = c("povodi","row_ker","col_ker","poc","roky_cal","iter","mae","rmse","nse","pi","x","id")
saveRDS(vysl,"D:/pokusydip/vysl.rds")
saveRDS(vysl_cr,"D:/pokusydip/vysl_cr.rds")

#########################################################################################x
#########################       CNN ON FLY          #####################################
#########################################################################################

{library(Rcpp)
  setwd("C:/Users/danek/Desktop/mlpR/vsnn")
  library(ggplot2)
  Rcpp::sourceCpp("rcppstuff.cpp")
  povodi <- list()
  minmax <- data.frame()
  for(i in 1:5){
    Qcamel <- read.table(paste0("./vyberdata/",i,"_q.txt"), header=FALSE)
    Qcamel <- Qcamel[!(Qcamel$V3 == 2 & Qcamel$V4 == 29), ]  
    Rcamel <- read.table(paste0("./vyberdata/",i,"_r.txt"), header=FALSE, skip = 4)
    Rcamel <- Rcamel[!(Rcamel$V2 == 2 & Rcamel$V3 == 29), ]  
    cur_pov <- data.frame(Datum = as.Date(paste(Rcamel$V1,Rcamel$V2,Rcamel$V3, sep = "-")),
                          Q = (Qcamel$V5 * 0.0283168466), R = Rcamel$V6, Tmax = Rcamel$V9)
    minmax[i,1] = min(cur_pov$Q)
    minmax[i,2] = max(cur_pov$Q)
    cur_pov$Q <- (cur_pov$Q - min(cur_pov$Q))/(max(cur_pov$Q)-min(cur_pov$Q))
    cur_pov$R <- (cur_pov$R - min(cur_pov$R))/(max(cur_pov$R)-min(cur_pov$R))
    cur_pov$Tmax <- (cur_pov$Tmax - min(cur_pov$Tmax))/(max(cur_pov$Tmax)-min(cur_pov$Tmax))
    povodi[[i]] <- cur_pov
    rm(cur_pov,Qcamel,Rcamel)
  }
  names(minmax) <- c("minQ","maxQ")
  names_povodi <- c("salmon","bigsur","merced","arroyo","andreas")
  names(povodi) <- names_povodi
  
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
      cit = cit + (obs[i] - mod[i])^2
      jme = jme + (obs[i] - mean(obs))^2
    }
    err = 1-(cit/length(mod))/(jme/length(mod))
    return(err)
  }
  
  pi <- function(mod, obs) {
    if (length(mod) != length(obs)) {
      stop("Vektory musí mít stejnou délku")
    }
    err = 0
    cit = 0
    jme = 0
    for (i in 2:length(mod)){
      cit = cit + (obs[i] - mod[i])^2
      jme = jme + (obs[i] - obs[i-1])^2
    }
    err = 1-(cit/length(mod))/(jme/length(mod))
    return(err)
  }
  
}  # nacteni Rcpp, dat a funkce kriterii

output_folder <- "D:/pokusydip/"
x = 1
vysl = data.frame()
vysl_cr = list()
for(n in c(2,3,5)){
for(m in c(3,5,7)){
for(l in c(10,30,70)){
for(k in c(10,20)){
for(j in c(10,50,100)){
for(i in 5){
  row_ker = n
  col_ker = m
  poc_ker = l
  roky_cal = k
  iter = j
  
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
  
  # Vytvoření datového rámce pro ggplot
  df_plot <- data.frame(
    Datum = povodi[[i]]$Datum[((roky_cal+row_ker-1)*365+1):length(povodi[[i]]$Datum)],
    Q_měřené = c(povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1+col_ker):length(povodi[[i]]$Q)],
                 povodi[[i]]$Q[((roky_cal+row_ker-1)*365+1):((roky_cal+row_ker-1)*365+col_ker)])*(minmax[i,2]-minmax[i,1])+minmax[i,1],
    Q_modelované = vystupy*(minmax[i,2]-minmax[i,1])+minmax[i,1]
  )
  
  # Vytvoření ggplot objektu
  plot <- ggplot(df_plot, aes(x = Datum)) +
    geom_line(aes(y = Q_měřené, color = "Měřené"), linewidth = 0.6) +
    geom_line(aes(y = Q_modelované, color = "Model"), linewidth = 0.45) +
    scale_color_manual(values = c("Měřené" = "black", "Model" = "red")) +
    labs(
      title = paste0("povodi=", i, " radky=", row_ker, " sloupce=", col_ker, 
                     " poc=", poc_ker, " roky cal=", roky_cal, " iter=", iter),
      subtitle = paste0(
        "mae=", round(mae(df_plot$Q_modelované, df_plot$Q_měřené),3), 
        "    rmse=", round(rmse(df_plot$Q_modelované, df_plot$Q_měřené),3), 
        "    nse=", round(nse(df_plot$Q_modelované, df_plot$Q_měřené),3), 
        "    pi=", round(pi(df_plot$Q_modelované, df_plot$Q_měřené),3)
      ),
      x = NULL,
      y = "Q [m3/s]",
      colour = NULL
    ) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 10)
    )
  
  cur_id = paste0(i,"_",row_ker,"_",col_ker,"_",poc_ker,"_",roky_cal,"_",iter,"_o")
  # Uložení grafu do souboru
  file_path <- file.path(output_folder, paste0(cur_id, ".png"))
  ggsave(file_path, plot = plot, width = 8, height = 6, dpi = 300, bg = "white")
  
  print(x)
  vysl[x,1] = names_povodi[i]
  vysl[x,2] = row_ker
  vysl[x,3] = col_ker
  vysl[x,4] = poc_ker
  vysl[x,5] = roky_cal
  vysl[x,6] = iter
  vysl[x,7] = round(mae(df_plot$Q_modelované,df_plot$Q_měřené),3)
  vysl[x,8] = round(rmse(df_plot$Q_modelované,df_plot$Q_měřené),3)
  vysl[x,9] = round(nse(df_plot$Q_modelované,df_plot$Q_měřené),3)
  vysl[x,10] = round(pi(df_plot$Q_modelované,df_plot$Q_měřené),3)
  vysl[x,11] = x
  vysl[x,12] = cur_id
  vysl_cr[[x]] <- df_plot
  x= x+1
}}}}}}
names(vysl) = c("povodi","row_ker","col_ker","poc","roky_cal","iter","mae","rmse","nse","pi","x","id")
saveRDS(vysl,"D:/pokusydip/vysl.rds")
saveRDS(vysl_cr,"D:/pokusydip/vysl_cr.rds")
andrvyslted[andrvyslted$pi>0,]
andrvyslted = readRDS("D:/pokusydip/pov1_1D_velic1/vysl.rds")

#########################################################################################x
#########################         1D CNN          #####################################
#########################################################################################

{library(Rcpp)
  setwd("C:/Users/danek/Desktop/mlpR/vsnn")
  library(ggplot2)
  Rcpp::sourceCpp("rcppstuff.cpp")
  povodi <- list()
  minmax <- data.frame()
  for(i in 1:5){
    Qcamel <- read.table(paste0("./vyberdata/",i,"_q.txt"), header=FALSE)
    Qcamel <- Qcamel[!(Qcamel$V3 == 2 & Qcamel$V4 == 29), ]  
    Rcamel <- read.table(paste0("./vyberdata/",i,"_r.txt"), header=FALSE, skip = 4)
    Rcamel <- Rcamel[!(Rcamel$V2 == 2 & Rcamel$V3 == 29), ]  
    cur_pov <- data.frame(Datum = as.Date(paste(Rcamel$V1,Rcamel$V2,Rcamel$V3, sep = "-")),
                          Q = (Qcamel$V5 * 0.0283168466), R = Rcamel$V6, Tmax = Rcamel$V9)
    minmax[i,1] = min(cur_pov$Q)
    minmax[i,2] = max(cur_pov$Q)
    cur_pov$Q <- (cur_pov$Q - min(cur_pov$Q))/(max(cur_pov$Q)-min(cur_pov$Q))
    cur_pov$R <- (cur_pov$R - min(cur_pov$R))/(max(cur_pov$R)-min(cur_pov$R))
    cur_pov$Tmax <- (cur_pov$Tmax - min(cur_pov$Tmax))/(max(cur_pov$Tmax)-min(cur_pov$Tmax))
    povodi[[i]] <- cur_pov
    rm(cur_pov,Qcamel,Rcamel)
  }
  names(minmax) <- c("minQ","maxQ")
  names_povodi <- c("salmon","bigsur","merced","arroyo","andreas")
  names(povodi) <- names_povodi
  
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
      cit = cit + (obs[i] - mod[i])^2
      jme = jme + (obs[i] - mean(obs))^2
    }
    err = 1-(cit/length(mod))/(jme/length(mod))
    return(err)
  }
  
  pi <- function(mod, obs) {
    if (length(mod) != length(obs)) {
      stop("Vektory musí mít stejnou délku")
    }
    err = 0
    cit = 0
    jme = 0
    for (i in 2:length(mod)){
      cit = cit + (obs[i] - mod[i])^2
      jme = jme + (obs[i] - obs[i-1])^2
    }
    err = 1-(cit/length(mod))/(jme/length(mod))
    return(err)
  }
  
}  # nacteni Rcpp, dat a funkce kriterii

output_folder <- "D:/pokusydip/"
x = 1
vysl = data.frame()
vysl_cr = list()
for(o in 1){
for(n in c(3,5,7,14)){
for(m in c(10,30,50)){
for(l in c(10,20)){
for(k in c(100,200,500)){
for(i in 4){

  ker = n
  poc_ker = m
  roky_cal = l
  iter = k
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
  
  
  # Vytvoření datového rámce pro ggplot
  df_plot <- data.frame(
    Datum = povodi[[i]]$Datum[(roky_cal * 365 + 1 + ker):length(povodi[[i]]$Datum)],
    Q_měřené = povodi[[i]]$Q[(roky_cal * 365 + 1 + ker):length(povodi[[i]]$Q)]*(minmax[i,2]-minmax[i,1])+minmax[i,1],
    Q_modelované = vystupy*(minmax[i,2]-minmax[i,1])+minmax[i,1]
  )
  
  # Vytvoření ggplot objektu
  plot <- ggplot(df_plot, aes(x = Datum)) +
    geom_line(aes(y = Q_měřené, color = "Měřené"), linewidth = 0.6) +
    geom_line(aes(y = Q_modelované, color = "Model"), linewidth = 0.45) +
    scale_color_manual(values = c("Měřené" = "black", "Model" = "red")) +
    labs(
      title = paste0("povodi=", i, " velic=", velic, " ker=", ker, 
                     " poc=", poc_ker, " roky cal=", roky_cal, " iter=", iter),
      subtitle = paste0(
        "mae=", round(mae(df_plot$Q_modelované, df_plot$Q_měřené),3), 
        "    rmse=", round(rmse(df_plot$Q_modelované, df_plot$Q_měřené),3), 
        "    nse=", round(nse(df_plot$Q_modelované, df_plot$Q_měřené),3), 
        "    pi=", round(pi(df_plot$Q_modelované, df_plot$Q_měřené),3)
      ),
      x = NULL,
      y = "Q [m3/s]",
      colour = NULL
    ) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 10)
    )
  
  cur_id = paste0(i,"_",velic,"_",ker,"_",poc_ker,"_",roky_cal,"_",iter)
  # Uložení grafu do souboru
  file_path <- file.path(output_folder, paste0(cur_id, ".png"))
  ggsave(file_path, plot = plot, width = 8, height = 6, dpi = 300, bg = "white")
  
  
  print(x)
  vysl[x,1] = names_povodi[i]
  vysl[x,2] = velic
  vysl[x,3] = ker
  vysl[x,4] = poc_ker
  vysl[x,5] = roky_cal
  vysl[x,6] = iter
  vysl[x,7] = round(mae(df_plot$Q_modelované,df_plot$Q_měřené),3)
  vysl[x,8] = round(rmse(df_plot$Q_modelované,df_plot$Q_měřené),3)
  vysl[x,9] = round(nse(df_plot$Q_modelované,df_plot$Q_měřené),3)
  vysl[x,10] = round(pi(df_plot$Q_modelované,df_plot$Q_měřené),3)
  vysl[x,11] = x
  vysl[x,12] = cur_id
  
  vysl_cr[[x]] <- df_plot
  x= x+1
}}}}}}
names(vysl) = c("povodi","velic","ker","poc","roky_cal","iter","mae","rmse","nse","pi","x","id")
saveRDS(vysl,"D:/pokusydip/vysl.rds")
saveRDS(vysl_cr,"D:/pokusydip/vysl_cr.rds")
vysl[vysl$pi == max(vysl$pi),]
#########################################################################################x
#########################         FULL CNN          #####################################
#########################################################################################

{library(Rcpp)
  setwd("C:/Users/danek/Desktop/mlpR/vsnn")
  library(ggplot2)
  Rcpp::sourceCpp("rcppstuff.cpp")
  povodi <- list()
  minmax <- data.frame()
  for(i in 1:5){
    Qcamel <- read.table(paste0("./vyberdata/",i,"_q.txt"), header=FALSE)
    Qcamel <- Qcamel[!(Qcamel$V3 == 2 & Qcamel$V4 == 29), ]  
    Rcamel <- read.table(paste0("./vyberdata/",i,"_r.txt"), header=FALSE, skip = 4)
    Rcamel <- Rcamel[!(Rcamel$V2 == 2 & Rcamel$V3 == 29), ]  
    cur_pov <- data.frame(Datum = as.Date(paste(Rcamel$V1,Rcamel$V2,Rcamel$V3, sep = "-")),
                          Q = (Qcamel$V5 * 0.0283168466), R = Rcamel$V6, Tmax = Rcamel$V9)
    minmax[i,1] = min(cur_pov$Q)
    minmax[i,2] = max(cur_pov$Q)
    cur_pov$Q <- (cur_pov$Q - min(cur_pov$Q))/(max(cur_pov$Q)-min(cur_pov$Q))
    cur_pov$R <- (cur_pov$R - min(cur_pov$R))/(max(cur_pov$R)-min(cur_pov$R))
    cur_pov$Tmax <- (cur_pov$Tmax - min(cur_pov$Tmax))/(max(cur_pov$Tmax)-min(cur_pov$Tmax))
    povodi[[i]] <- cur_pov
    rm(cur_pov,Qcamel,Rcamel)
  }
  names(minmax) <- c("minQ","maxQ")
  names_povodi <- c("salmon","bigsur","merced","arroyo","andreas")
  names(povodi) <- names_povodi
  
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
      cit = cit + (obs[i] - mod[i])^2
      jme = jme + (obs[i] - mean(obs))^2
    }
    err = 1-(cit/length(mod))/(jme/length(mod))
    return(err)
  }
  
  pi <- function(mod, obs) {
    if (length(mod) != length(obs)) {
      stop("Vektory musí mít stejnou délku")
    }
    err = 0
    cit = 0
    jme = 0
    for (i in 2:length(mod)){
      cit = cit + (obs[i] - mod[i])^2
      jme = jme + (obs[i] - obs[i-1])^2
    }
    err = 1-(cit/length(mod))/(jme/length(mod))
    return(err)
  }
  
}  # nacteni Rcpp, dat a funkce kriterii

output_folder <- "D:/pokusydip/"
x = 1
vysl = data.frame()
vysl_cr = list()
for(i in 3){
  iter = 5
  
  mlp <- udelej_nn()
  nn_init_nn(mlp,50,c(50,50,20))
  nn_set_vstup_rady(mlp, povodi[[i]]$Q[1:1000],
                    povodi[[i]]$Q[1001:12000], 
                    povodi[[i]]$R[1:1000],
                    povodi[[i]]$R[1001:12000], 
                    povodi[[i]]$Tmax[1:1000],
                    povodi[[i]]$Tmax[1001:12000]
  )
  nn_cnn_full_cal(mlp,iter)
  nn_cnn_full_val(mlp)
  vystupy <- nn_get_vystupy(mlp)

  # Vytvoření datového rámce pro ggplot
  df_plot <- data.frame(
    Datum = povodi[[i]]$Datum[1421:12020],
    Q_měřené = povodi[[i]]$Q[1421:12020]*(minmax[i,2]-minmax[i,1])+minmax[i,1],
    Q_modelované = vystupy*(minmax[i,2]-minmax[i,1])+minmax[i,1]
  )
  
  # Vytvoření ggplot objektu
  plot <- ggplot(df_plot, aes(x = Datum)) +
    geom_line(aes(y = Q_měřené, color = "Měřené"), linewidth = 0.6) +
    geom_line(aes(y = Q_modelované, color = "Model"), linewidth = 0.45) +
    scale_color_manual(values = c("Měřené" = "black", "Model" = "red")) +
    labs(
      title = paste0("povodi=", i, " iter=", iter),
      subtitle = paste0(
        "mae=", round(mae(df_plot$Q_modelované, df_plot$Q_měřené),3), 
        "    rmse=", round(rmse(df_plot$Q_modelované, df_plot$Q_měřené),3), 
        "    nse=", round(nse(df_plot$Q_modelované, df_plot$Q_měřené),3), 
        "    pi=", round(pi(df_plot$Q_modelované, df_plot$Q_měřené),3)
      ),
      x = NULL,
      y = "Q [m3/s]",
      colour = NULL
    ) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 10)
    )
  
  cur_id = paste0(i,"_",iter,"_f")
  # Uložení grafu do souboru
  file_path <- file.path(output_folder, paste0(cur_id, ".png"))
  ggsave(file_path, plot = plot, width = 8, height = 6, dpi = 300, bg = "white")
  
  print(x)
  vysl[x,1] = i
  vysl[x,2] = iter
  vysl[x,3] = round(mae(df_plot$Q_modelované,df_plot$Q_měřené),3)
  vysl[x,4] = round(rmse(df_plot$Q_modelované,df_plot$Q_měřené),3)
  vysl[x,5] = round(nse(df_plot$Q_modelované,df_plot$Q_měřené),3)
  vysl[x,6] = round(pi(df_plot$Q_modelované,df_plot$Q_měřené),3)
  vysl[x,7] = x
  vysl[x,8] = cur_id
  vysl_cr[[x]] <- df_plot
  x= x+1
}
names(vysl) = c("povodi","iter","mae","rmse","nse","pi","x","id")
saveRDS(vysl,"D:/pokusydip/vysl.rds")
saveRDS(vysl_cr,"D:/pokusydip/vysl_cr.rds")

