# parameters for the l2l-comet-[..]-ga.py script
# BRUNEL MODEL

# Optimizee params
optimizee_params = {
    'seed': 123,
    'keys_to_evolve': ['P', 'eta'],
    'threads': 8
}

# Outer-loop optimizer initialization
optimizer_params = {
    'seed': 123,
    'popsize': 48,
    'CXPB': 0.8,
    'MUTPB': 0.2,
    'NGEN': 100,
    'indpb': 0.2,
    'tournsize': 3,
    'matepar': 0.5,
    'mutpar': 0.05
}
