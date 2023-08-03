import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model2 import Conv_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.min_epsilon = 2
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        #self.model = Linear_QNet()
        self.model = Conv_QNet((3, 24, 32), 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head_channel = np.zeros((24, 32), dtype=int)
        body_channel = np.zeros((24, 32), dtype=int)
        food_channel = np.zeros((24, 32), dtype=int)
        head = game.snake[0]
        head_channel[int(head.y - 20) // 20][int(head.x - 20) // 20] = 1 # Snake's head
        for point in game.snake[1:]: # Start from 1 to skip the head
            body_channel[int(point.y - 20) // 20][int(point.x - 20) // 20] = 1 # Snake's body
        food_channel[int(game.food.y - 20) // 20][int(game.food.x - 20) // 20] = 1 # Food location
        state = np.stack((head_channel, body_channel, food_channel), axis=0)
        state = np.expand_dims(state, axis=0)
        #print(f'State Shape at the end of the get_state method: {state.shape}')
        return state



    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        #print(f'States shape before long memory training: {states}')

        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = self.epsilon = max(80 - self.n_games, self.min_epsilon)
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            #print(f'State0: {state0}')
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        #print('Getting old state')
        state_old = agent.get_state(game)

        #print('Got old state. Getting move.')

        # get move
        final_move = agent.get_action(state_old)

        #print('Got move. Getting ')

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()