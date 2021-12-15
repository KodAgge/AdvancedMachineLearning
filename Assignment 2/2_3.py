import numpy as np
from scipy.stats import norm
from scipy.stats import gamma
import matplotlib.pyplot as plt


def calcMu(a_N, b_N, x, mu0, lambda0):
  N = len(x)                # N
  x_mean = np.mean(x)       # Mean of x
  tau_expected = a_N / b_N  # E[tau]

  mu_N = (lambda0 * mu0 + N * x_mean) / (lambda0 + N)
  lambda_N = (lambda0 + N) * tau_expected

  return mu_N, lambda_N


def calcTau(mu_N, lambda_N, x, a0, b0, mu0, lambda0):
  N = len(x)                                      # N
  x_mean = np.mean(x)                             # Mean of x
  x_square = sum(x ** 2)                          # sum(x^2)
  mu_expected = mu_N                              # E[mu]
  mu_square_expected = 1 / lambda_N + mu_N ** 2   # E[mu^2]

  a_N = a0 + N/2
  b_N = b0 + (1/2) * (x_square + lambda0 * mu0 ** 2) - (mu0 * lambda0 + N * x_mean) * mu_expected + (1/2) * (lambda0 + N) * mu_square_expected

  return a_N, b_N


def getData(n, mu, std_2, seed = 100):
  np.random.seed(seed)
  return np.random.normal(mu, std_2, n) # Drawing from gaussian distrbution


def threshold(a_N, b_N, mu_N, lambda_N, a_N_next, b_N_next, mu_N_next, lambda_N_next, threshold):
  # Checking if the change in variables are within the threshold
  if abs(a_N_next / a_N - 1) < threshold:
    if abs(b_N_next / b_N - 1) < threshold:
      if abs(mu_N_next / mu_N - 1) < threshold:
        if abs(lambda_N_next / lambda_N - 1) < threshold:
          return False
  return True


def printVIResults(data, iteration):
  labels = ["a", "b", "mu", "lambda"]
  for i in range(len(labels)):
    print(labels[i] + " | From " + str(data[i]) + " to " + str(data[i+4]))
  print("It took " + str(iteration) + " iterations.")


def printTrueResults(data):
  labels = ["a", "b", "mu", "beta"]
  for i in range(len(labels)):
    print(labels[i] + " | " + str(data[i]) )


def q_mu(mus, mu, lambd):
  return norm.pdf(mus, mu, np.sqrt(1 / lambd)) # Gaussian pdf


def q_tau(taus, a, b):
  return gamma.pdf(taus, a, loc = 0, scale = 1 / b) # Gamma pdf


def calcNormGamParam(x, mu_0, lambda_0, a0, b0):
  N = len(x)              # N
  x_mean = np.mean(x)     # Mean of x
  x_square = sum(x ** 2)  # sum(x^2)

  beta = N + lambda_0     # Without tau
  mu = (N * x_mean + lambda_0 * mu_0) / beta

  a = a0 + N / 2
  b = b0 + (lambda_0 * (mu_0 ** 2) - beta * (mu ** 2) + x_square) / 2

  return beta, mu, a, b


def relativeErrors(vi_results, true_results):
  labels = ["a", "b", "mu"]
  for i in range(len(labels)):
    print(labels[i] + " | " + str(round(100*abs(vi_results[i]/true_results[i]-1),2)) + "%" + " error")


def vi_algorithm(a_N, b_N, mu_N, lambda_N, x, mu0, lambda0, a0, b0):
  thresh = 1e-7

  running = True
  iteration = 0

  # Running algorithm
  while running:
    #Calculating new parameters
    mu_N_next, lambda_N_next = calcMu(a_N, b_N, x, mu0, lambda0)
    a_N_next, b_N_next = calcTau(mu_N, lambda_N, x, a0, b0, mu0, lambda0)

    #Checking if threshold is satisfied
    running = threshold(a_N, b_N, mu_N, lambda_N, a_N_next, b_N_next, mu_N_next, lambda_N_next, thresh)

    # Saving parameters
    a_N, b_N, mu_N, lambda_N = a_N_next, b_N_next, mu_N_next, lambda_N_next
    iteration += 1
  
  return a_N, b_N, mu_N, lambda_N, iteration


def pTrue(x, y, beta, mu, a, b):
  return norm.pdf(x, mu, np.sqrt(1 / (beta * y))) * gamma.pdf(y, a, loc = 0, scale = 1 / b) # Normal-gamma pdf


def plotResults(a, b, mu, precision, center, std_2, exact = False):
  # Getting interval that fits data
  mus = np.linspace(-4,4,300) * (std_2 ** (0.7)) + center
  taus = np.linspace(-0.1,6,200) * std_2 ** (-2)
  # mus = np.linspace(8.5,11.5,300)
  # taus = np.linspace(-0.1,15,300)
  # mus = np.linspace(9.9,10.05,300)
  # taus = np.linspace(3,4,300)

  # For the exact posterior
  if exact:
    color = "blue"
    Ms, Ts = np.meshgrid(mus, taus, indexing="ij")
    Z = np.zeros_like(Ms)

    for i in range(Z.shape[0]):
      for j in range(Z.shape[1]):
          Z[i][j] = pTrue(mus[i], taus[j], precision, mu, a, b)
  
  # The posterior calculated by the VI algorithm
  else:
    color = "red"
    q_mus = q_mu(mus, mu, precision)
    q_taus = q_tau(taus, a, b)
    Ms, Ts = np.meshgrid(mus, taus, indexing="ij")

    Z = np.outer(q_mus, q_taus)
  
  # Plotting the contour
  plt.contour(Ms, Ts, Z, 10, colors = color)


def main():
  # Data attributes
  n = 10000
  center = -2
  std = 30

  print("1. Simulating guassian data with\nmu =", center, "| sigma =", std, "| n =", n)

  x = getData(n, center, std, 1021)

  # -----------------------------------------------------------------------------------------------------

  a0, b0 = 0.01, 9
  mu0, lambda0 = -2, 10

  print("\n2. Setting prior parameters\na0 =", a0, "| b0 =", b0, "| mu0 =", mu0, "| lambda0 =", lambda0)

  # -----------------------------------------------------------------------------------------------------

  a_start, b_start = 1e-9, 1e-9
  mu_start, lambda_start = 1e-9, 1e-9

  print("\n3. Setting start values\na_start =", a_start, "| b_start =", b_start, "| mu_start =", mu_start, "| lambda_start =", lambda_start)

  a_N, b_N = a_start, b_start
  mu_N, lambda_N = mu_start, lambda_start

  # -----------------------------------------------------------------------------------------------------

  print("\n4. Running VI algorithm...")

  a_N, b_N, mu_N, lambda_N, iteration = vi_algorithm(a_start, b_start, mu_start, lambda_start, x, mu0, lambda0, a0, b0)

  printVIResults([a_start, b_start, mu_start, lambda_start, a_N, b_N, mu_N, lambda_N], iteration)

  # -----------------------------------------------------------------------------------------------------

  print("\n5. Calculating true posterior...")

  beta, mu, a, b = calcNormGamParam(x, mu0, lambda0, a0, b0)

  printTrueResults([a, b, mu, beta])

  # -----------------------------------------------------------------------------------------------------

  print("\n6. Calculate relative errors")

  relativeErrors([a_N, b_N, mu_N], [a, b, mu])

  # -----------------------------------------------------------------------------------------------------

  print("\n7. Plot results")  

  plotResults(a_N, b_N, mu_N, lambda_N, center, std)

  plotResults(a, b, mu, beta, center, std, True)

  plt.show()


main()
