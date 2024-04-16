import yaml
from accelerate.state import AcceleratorState
import accelerate

from omegaconf import OmegaConf

def load_config(file_path):
    """
    Load a YAML configuration file using OmegaConf.
    """
    config = OmegaConf.load(file_path)
    return config

def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]
    
