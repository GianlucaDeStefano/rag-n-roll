import hydra


def get_hydra_output_dir():
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    return hydra_cfg['runtime']['output_dir']

def get_hydra_mode():
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    return str(hydra_cfg['mode']).split('.')[1]

def get_hydra_sweep_dir():
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    return hydra_cfg['sweep']['dir']