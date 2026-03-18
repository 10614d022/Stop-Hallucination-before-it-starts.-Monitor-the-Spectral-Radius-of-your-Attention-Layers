def compute_tau_from_entropy(entropy: float, theta: float, alpha: float) -> float:
    if entropy >= theta:
        return 1.0
    return 1.0 + alpha * (theta - entropy)


def compute_tau_from_rho(rho: float, theta: float, alpha: float) -> float:
    if rho <= theta:
        return 1.0
    return 1.0 + alpha * (rho - theta)
