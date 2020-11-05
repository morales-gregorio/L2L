# parameters for the l2l-comet-[..]-ga.py script
# MICROCIRCUIT MODEL

# TODO update the keys to evolve !

seed = 123

# Optimizee params
optimizee_params = {
    'seed': seed,
    'keys_to_evolve': ['conn_probs', 'K_ext', 'bg_rate']
}

# Outer-loop optimizer initialization
optimizer_params = {
    'seed': seed,
    'popsize': 24,
    'CXPB': 0.8,
    'MUTPB': 0.002,
    'NGEN': 50,
    'indpb': 0.2,
    'tournsize': 3,
    'matepar': 0.5,
    'mutpar': 0.05
}
