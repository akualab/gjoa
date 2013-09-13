# Testing Results for  hmm.go 

# HMM lib
library(RHmm)

# observations
obs <- c(0.1, 0.3, 1.1, 1.2, 0.7, 0.7, 5.5, 7.8, 10.0, 5.2, 1.1, 1.3)

# The hmm has 2 states with Gaussians mean=1,var=1 and mean=4,var=4.
# initial state probs: 0.8, 0.2
# state trans prob: {{0.9, 0.1}, {0.3, 0.7}}

# initializing the model for RHmm package
init.p <- c(0.8, 0.2)
normal <- distributionSet(dis="NORMAL", mean=c(1, 4), var=c(1, 4))
transMat <- rbind(c(0.9, 0.1), c(0.3, 0.7)) 
hmm.mod <- HMMSet(init.p, transMat, normal) 

# calling forwardBackward
fb <- forwardBackward(hmm.mod, obs, logData=TRUE)

# alpha
alpha <- fb$Alpha
write.csv(alpha, "alpha.csv", row.names = FALSE)

ww <- w[,c(1,3,2,4)]
# beta
beta <- fb$Beta
write.csv(beta, "beta.csv", row.names = FALSE)

# gamma
gamma <- log(fb$Gamma)
write.csv(gamma, "gamma.csv", row.names = FALSE)

# xsi
xsi <- fb$Xsi
## transforming xsi
## Xsi is in a list format transforming to a matrix
w <- do.call(rbind,lapply(xsi, function(x) {log(as.vector(t(x)))}))
w[12, ] <- 0
w <- as.vector(w)

# making it pretty
Xsi <- t(matrix(w, ncol=12))
write.csv(Xsi, "xsi.csv", row.names = FALSE)

# viterbi 
v <- viterbi(hmm.mod, obs)
v$states -1
# my own implementation of alpha to test forwardBackward
m <- c(1, 4)
var <- c(1, 4)

log_density <- function(x, m, v) {
  s <- sqrt(v)
  dnorm(x, m, s, log=TRUE)
}

computeAlpha <- function(obs, transMat, init.p, m, var) {
  mm <- matrix(0, length(obs), length(init.p))
  mm[1,] <- log(init.p) + log_density(obs[1], m, var)
  for (i in 2:length(obs)) {
    summ <- t(t(transMat)%*%(exp(mm[i-1,])))
	mm[i,] <- log(summ) + log_density(obs[i], m, var)
  }
  mm
}

mm <- computeAlpha(obs, transMat, init.p, m, var)

gamma <- function(alpha, beta) {
	norm_constant <-  log(rowSums(exp(alpha + beta)))
	maxtrix.const <- matrix(rep(norm_constant,2), ncol=2)
	alpha + beta - maxtrix.const
}

