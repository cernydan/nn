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
nn_init_nn(mlp)
nn_online_bp(mlp,50000)
simulout <- nn_get_vystupy(mlp)
nn_set_valdata(mlp,dt2)
nn_valid(mlp)
simulout2 <- nn_get_vystupy(mlp)

plot(c(Qkal,Qval),type = "l")
lines(c(simulout,simulout2),col = "red")
error <- nn_count_cost(mlp)
error

