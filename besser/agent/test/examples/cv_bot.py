from besser.agent.core.scenario.scenario_image_object import ScenarioImageObject
from besser.agent.core.scenario.scenario_image_property import ScenarioImageProperty


import logging

from besser.agent.core.agent import Agent
from besser.agent.core.session import Session
from besser.agent.cv.object_detection.yoloworld_object_detector import YOLOWorldObjectDetector
from besser.agent.cv.vllm.vllm_openai_api import VLLMOpenAI
from besser.agent.core.scenario.scenario import and_ex, or_ex, not_ex

# Configure the logging module
logging.basicConfig(level=logging.INFO, format='{levelname} - {asctime}: {message}', style='{')

# Create the agent
agent = Agent('greetings_agent')
# Load agent properties stored in a dedicated file
agent.load_properties('config.ini')
# Define the platform your agent will use
websocket_platform = agent.use_websocket_platform(use_ui=True, video_input=True)


# Image objects

person = agent.new_image_object('person')
phone = agent.new_image_object('phone')
bottle = agent.new_image_object('bottle')

# Image properties
indoors = agent.new_image_property(name='indoors', description='The picture is taken in an indoor environment (i.e., not outdoors)')
iphone = agent.new_image_property(name='iphone', description='The phone in the image (if there is one) is an iPhone (Apple)')

yolo_model = YOLOWorldObjectDetector(agent=agent, name='yolov8l-worldv2', model_path='yolo_weights/yolov8l-worldv2.pt', parameters={
    'classes': [image_object.name for image_object in agent.image_objects]
})
vllm = VLLMOpenAI(agent, 'gpt-4o', {})

scenario = agent.new_scenario('scenario1')
scenario.set_expression(
    and_ex([
        ScenarioImageProperty('iphone', iphone, score=0.5),  # LINK TO OBJECT
        ScenarioImageObject('person', person, max=3, score=0.3),  # choose which model to use at property level or scenario level
        ScenarioImageObject('phone', phone, 0.5)
    ])
)
initial_state = agent.new_state('initial_state', initial=True)
person_state = agent.new_state('person_state')


initial_state.when_scenario_matched_go_to(scenario, person_state)  # transitions not part of the DSL

# DSL: Scenarios and impl config (llms, models...)


def hello_body(session: Session):
    session.reply('Hi!')


def person_body(session: Session):
    session.reply('I can see you!')


person_state.set_body(person_body)
person_state.when_no_intent_matched_go_to(initial_state)

# RUN APPLICATION

if __name__ == '__main__':
    agent.run()
