from omegaconf import OmegaConf
import numpy as np

def register_custom_resolvers():
	OmegaConf.register_new_resolver("log_space", lambda x, y, z: ",".join(map(str,np.logspace(x,y,z,dtype=int))))
	OmegaConf.register_new_resolver("eval", eval)
	OmegaConf.register_new_resolver("int", int)
	OmegaConf.register_new_resolver("sqrt", np.sqrt)
	return

register_custom_resolvers()