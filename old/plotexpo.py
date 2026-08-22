import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc


# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
mean_sig_p = 5.28   # mu
sigma_sig_p = 0.03  # sigma
b_t_p = 10.0       # exponential slope


# ------------------------------------------------------------
# Function corresponding to the RooGenericPdf
# ------------------------------------------------------------
def tail_plus(m, b, mean, sigma):
    return np.exp(b * (m - mean)) * erfc((m - mean) / sigma)


# ------------------------------------------------------------
# x range
# ------------------------------------------------------------
m = np.linspace(5.0, 5.6, 1000)


# ------------------------------------------------------------
# Evaluate function
# ------------------------------------------------------------
y = tail_plus(
    m,
    b_t_p,
    mean_sig_p,
    sigma_sig_p
)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))

plt.plot(m, y, label="tail_plus")

plt.axvline(
    mean_sig_p,
    color="red",
    linestyle="--",
    alpha=0.7,
    label=r"$\mu$"
)

plt.xlabel(r"$m$")
plt.ylabel(r"$f(m)$")
plt.title(r"$e^{b(m-\mu)}\,\mathrm{erfc}\left((m-\mu)/\sigma\right)$")

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("/eos/user/a/ahulsber/scripts/scripts/plotexpo.pdf")
