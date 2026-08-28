from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(config_path="conf", config_name="base_config", version_base=None)
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()