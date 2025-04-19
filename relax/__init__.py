try:
    import relax_env
    del relax_env
except ImportError:
    print("Cannot import relax_env, additional environments will not be available.")
