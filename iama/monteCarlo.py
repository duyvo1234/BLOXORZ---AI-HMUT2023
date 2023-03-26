from .node import Node
from collections import defaultdict
import random
import numpy as np

class MonteCarloTreeSearch(object):

    def __init__(self, node, problem):
        self.root = node
        self.problem = problem
        self.root._update_untried_children(problem)

    def best_action(self, simulations_number=None):
        for _ in range(0, simulations_number):
            visited = []
            trvNode = self.root
            # selection
            while trvNode._untried_children == [] and trvNode.children != []:
                trvNode = trvNode.bestChild()
                trvNode._update_untried_children(self.problem)
            # expansion
            if trvNode._untried_children != []:
                m = random.choice(trvNode._untried_children)
                trvNode = trvNode.add_child(m)
            # rollout
            should_stop = False
            while trvNode.expand(self.problem) != [] and not should_stop:
                choices = list(set(trvNode.expand(self.problem)) - set(visited))
                if choices != []:
                    trvNode = random.choice(choices)
                    visited.append(trvNode)
                else:
                    should_stop = True
            
            prvNode = self.problem.h1(trvNode)
            # backpropagate
            while trvNode != None:
                result = self.problem.h1(trvNode)
                trvNode.backpropagate(-result)
                # prvNode = result
                trvNode = trvNode.parent
            
        # select most visited child
        return sorted(self.root.children, key = lambda c: c.visits)[-1]

class MonteNode(Node):
    def __init__(self,state, parent=None, action=None, path_cost=0):
        super().__init__(state, parent, action, path_cost)
        self._untried_children = None
        self.children = []
        self.wins = 0
        self.visits = 0

    def _update_untried_children(self, problem):
        self._untried_children = self.expand(problem)

    def bestChild(self):
        s = sorted(self.children, key = lambda c: c.wins/c.visits + np.sqrt(2*np.log(self.visits)/c.visits))[-1]
        return s
    
    def add_child(self, child):
        self.children.append(child)
        self._untried_children.remove(child)
        return child

    def expand(self, problem):
        return [self._child_node(problem, action)
        for action in problem.actions(self.state)]

    def _child_node(self, problem, action):
        next_state = problem.result(self.state, action)
        next_node = MonteNode(next_state, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next_state))
        return next_node
    
    def backpropagate(self, result):
        self.wins += result
        self.visits += 1