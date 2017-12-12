from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer
from os.path import join

root_dir = "/home/icub/user_files/BBC_Demo/rasa_files"
data_file = "demo-rasa.json"
model_dir = "models"
config_file = "config_spacy.json"

rasa_config = RasaNLUConfig(join(root_dir, config_file))

training_data = load_data(join(root_dir, data_file))
trainer = Trainer(rasa_config)
trainer.train(training_data)
mod = trainer.persist(join(root_dir, model_dir))
print mod
