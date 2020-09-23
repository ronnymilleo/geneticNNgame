from gann import GeneticANN


class Player:
    """Player's class"""

    def __init__(self, pos_x, pos_y):
        self.pos_x = pos_x//64 + 1
        self.pos_y = 10 - pos_y//64
        self.cur_step = [0]
        self.step_list = []
        self.neural_network = GeneticANN()

    def save_play(self, step, dist):
        self.step_list.append((step, self.pos_x, self.pos_y, dist))

    def status(self):
        print('Step = {} - Pos = ({}, {})'.format(self.cur_step, self.pos_x, self.pos_y))
        print('Fitness = {}'.format(self.neural_network.fitness))
