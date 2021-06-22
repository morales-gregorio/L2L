# parameters for the l2l-comet-[..]-ga.py script
# BRUNEL MODEL

# Optimizee params
optimizee_params = {
    'seed': 123,
    'keys_to_evolve': ['P'],
    'threads': 4
}

# Outer-loop optimizer initialization
optimizer_params = {
    'seed': 1234,
    'pop_size': 36,
    'cx_prob': 0.8,
    'mut_prob': 0.2,
    'n_iteration': 100,
    'ind_prob': 0.2,
    'tourn_size': 3,
    'mut_par': 0.05,
    'mate_par': 0.5
}
