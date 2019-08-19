from __future__ import absolute_import
from __future__ import division
from __future__ import  unicode_literals

from rasa.core.actions.action import Action
from rasa.core.events import SlotSet

class ActionWeather(Action):
    def name(self):
        return 'action_weather'

