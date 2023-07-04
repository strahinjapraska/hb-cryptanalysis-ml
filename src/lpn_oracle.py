import numpy as np

class LPNOracle:
    def __init__(self, secret, error_rate):
        self.secret = secret
        self.dimension = len(secret)
        self.error_rate = error_rate

    def sample(self, n_amount):
        # Create random matrix.
        A = np.random.randint(0, 2, size=(n_amount, self.dimension))
        # Add Bernoulli errors.
        e = np.random.binomial(1, self.error_rate, n_amount)
        # Compute the labels.
        b = np.mod(A @ self.secret + e, 2)
        return A, b
    
def calculate_threshold(k,tau,m):
    return m*tau + np.sqrt(k*m)

def check_prediction(A,b,s_prime,t):
    return np.mod(A@s_prime+b,2).sum() <= t 


