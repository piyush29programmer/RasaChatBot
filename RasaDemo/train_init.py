from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import asyncio
import logging
from typing import Text
from rasa.nlu.model import Trainer
from rasa.nlu.training_data import load_data
from rasa.nlu import config
from rasa.core.agent import Agent
from rasa.core.policies.keras_policy import KerasPolicy
from rasa.core.policies.memoization import MemoizationPolicy
import os

logfile = 'Rasa.log'


async def train_nlu(data_path, configs, model_path):
    logging.basicConfig(filename=logfile, level=logging.DEBUG)
    training_data = load_data(data_path)
    trainer = Trainer(config.load(configs))
    trainer.train(training_data)
    trainer.persist(model_path, fixed_model_name='nlu')


async def train_core(domain_file: Text = "domain.yml",
                     model_directory: Text = "models",
                     model_name: Text = "core",
                     training_data_file: Text = "data/stories.md",):

    logging.basicConfig(filename=logfile, level=logging.DEBUG)

    agent = Agent(
        domain_file,
        policies=[
            MemoizationPolicy(max_history=3),
            KerasPolicy(batch_size=100, epochs=400, validation_split=0.2)
        ]
    )
    training_data = await agent.load_data(training_data_file)
    agent.train(training_data)

    # Attention: agent.persist stores the model and all meta data into a folder.
    # The folder itself is not zipped.
    model_path = os.path.join(model_directory, model_name)
    agent.persist(model_path)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(train_nlu('./data/data.json', 'nlu_config.yml', './models'))
    loop.run_until_complete(train_core())



