import yaml

model_config_path = "/lustre/fs01/portfolios/dir/users/jeromek/savanna-context-ext/configs/7b-context-ext/model_configs/7b_stripedhyena2_base_4M_32k.yml"

def read_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config = read_config(model_config_path)
    print(config['operator-config'])
    print(len(config['operator-config']))