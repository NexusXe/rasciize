import numpy as np
from numpy.polynomial import Polynomial, Chebyshev

u = np.linspace(0, np.pi, 100000000)
target = np.sinc(np.sqrt(u)) # np.sinc includes the pi factor

cheb_fit = Chebyshev.fit(u, target, deg=6)
poly_fit = cheb_fit.convert(kind=Polynomial)
coeffs = poly_fit.coef

approx = poly_fit(u)
error = np.max(np.abs(approx - target))

print(f"Max Error: {error:.2e}")
print("-" * 30)
print("Standard Polynomial Coefficients (Low to High Degree):")
print(f"[{', '.join(f'{repr(c.astype(np.float32))}' for c in coeffs)}]")
