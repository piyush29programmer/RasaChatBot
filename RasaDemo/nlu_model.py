import logging
import pprint
from rasa.nlu.training_data import load_data
from rasa.nlu import config
from rasa.nlu.model import Trainer
from rasa.nlu.model import Interpreter
from rasa.core.validator import Validator

logfile = 'nlu_model.log'


def train_nlu(data_path, configs, model_path):
    logging.basicConfig(filename=logfile, level=logging.DEBUG)
    training_data = load_data(data_path)
    trainer = Trainer(config.load(configs))
    trainer.train(training_data)
    model_directory = trainer.persist(model_path, fixed_model_name='nlu')


def run_nlu():
    interpreter=Interpreter.load('./models/nlu')
    print(interpreter.parse("I am planning holiday to india"))


if __name__ == '__main__':
    train_nlu('./data/data.json', 'nlu_config.yml', './models')
    run_nlu()