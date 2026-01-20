import numpy as np
import numpy.polynomial.polynomial as poly

def normalized_sinc(x):
    """
    Computes sin(pi * x) / (pi * x).
    Handles x=0 gracefully.
    """
    # Avoid division by zero by using np.sinc which is defined as sin(pi*x)/(pi*x)
    return np.sinc(x)

def generate_coefficients():
    print("Generating coefficients for Normalized Sinc approximation...")
    print("Optimization: Approximating f(y) = sinc(sqrt(y)) where y = x^2")

    x_min = 0.0
    x_max = np.pi
    
    # We use a degree 10 polynomial in y (equivalent to degree 20 in x).
    # This provides a balance between high accuracy and instruction count.
    # Degree 10 allows us to capture the 5 roots and the decaying amplitude effectively.
    DEGREE = 10
    
    # Define the domain for y = x^2
    y_min, y_max = x_min**2, x_max**2
    
    # 1. Generate Chebyshev nodes (sampling points) for better stability than equidistant points
    # These are roots of Chebyshev polynomial of degree (DEGREE + 1), mapped to [0, 25]
    k = np.arange(1, DEGREE + 2)
    nodes_cheb = np.cos((2 * k - 1) / (2 * (DEGREE + 1)) * np.pi)
    # Map from [-1, 1] to [y_min, y_max]
    y_nodes = 0.5 * (nodes_cheb + 1) * (y_max - y_min) + y_min
    
    # 2. Calculate target values
    # Note: we are calculating sinc(sqrt(y))
    target_values = normalized_sinc(np.sqrt(y_nodes))
    
    # 3. Fit the polynomial
    # We fit y_nodes vs target_values directly using a Polynomial basis
    # (Least squares fit on Chebyshev nodes is a near-minimax approximation)
    coefs = poly.polyfit(y_nodes, target_values, DEGREE)
    
    # 4. Output Rust Code
    print(f"// Degree {DEGREE} polynomial in x^2 (Degree {DEGREE*2} in x)")

    for i, c in enumerate(coefs):
        print(f"const C{i}: f32 = {repr(np.float32(c))[11:-1]};") # Emit only as much precision as needed to exactly match the float32 representation
    
    # Calculate Max Error for verification
    test_x = np.linspace(x_min, x_max, 5000)
    test_y = test_x**2
    exact = normalized_sinc(test_x)
    approx = poly.polyval(test_y, coefs)
    error = np.abs(exact - approx)
    print(f"\nMax Absolute Error: {np.max(error):.2e}")

if __name__ == "__main__":
    generate_coefficients()