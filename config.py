import yaml

class Config:
    
    with open("configs/config.yaml") as f:
        _CONFIG = yaml.safe_load(f)

    @classmethod
    def get(self, key: str):
        
        config = self._CONFIG
        
        for k in key.split("/"):
            config = config.get(k)
            if config is None:
                return None

        return config