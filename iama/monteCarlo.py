from .node import Node
from collections import defaultdict
import random
import numpy as np

class MonteNode(Node):
    def __init__(self, state, parent=None, action=None, path_cost=0):
        super().__init__(state, parent, action, path_cost)
        self.number_of_visit = 0
        self.value = 0
        self.tried_state = []
        self.child = []

    def is_fully_expanded(self):
        return len(self.untried_move) == 0
    
    def get_child(self, problem):
        self.child = self.expand(problem)

    def is_state_tried(self, state):
        return state in self.tried_state
    
    def rollout(self, problem, state):
        rollout_node = MonteNode(state)
        while not problem.goal_test(rollout_node) and not self.is_state_tried(rollout_node.state):
            rollout_node.get_child(problem)
            rollout_node = random.choice(rollout_node.child)

        if problem.goal_test(rollout_node):
            return 1
        else:
            return 0
    
    def backpropagate(self, result):
        while self.parent != None:
            self.value += result
            self.number_of_visit += 1
            self.parent.backpropagate(result)

    def bestChild(self, c_param=1.4):
        choices = [
            (c.value/c.number_of_visit) + c_param * np.sqrt((2*np.log(self.number_of_visit) / c.number_of_visit))
            for c in self.child
        ]
        return self.child[np.argmax(choices)]



    
    
    

