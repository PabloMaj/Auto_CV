
from src.logger import log

def evaluator_agent(state):
    code = '''
def evaluate(gt, pred):
    return abs(gt - pred)
'''

    state["evaluator_code"] = code

    log(state, "Evaluator generated")

    return state
