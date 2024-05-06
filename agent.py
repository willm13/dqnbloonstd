'''
DQN main model/target model training largely from here:
https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/
'''

import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Concatenate
import keras
import random
import numpy as np
from collections import deque
import os
import time
from GameBrowser import Browser
import pickle

INPUT_SIZE = 595 # 3 (money, round, lives) + 74 x 6 (74 positions, one-hot encoding of size 6) + 74 x 2 (79 possible towers, 2 upgrades each)
NUM_POS = 74 # number of positions = 79
ACTION_SIZE = 593


# handle load model or create new model
def init():
    if os.path.exists('models/model.keras'):
        print("!!!!!!!!!!!! Hello, loading model")
        model = load_model('models/model.keras')
    else:
        print("Hello, new model")
        input_layer = keras.Input(shape=(INPUT_SIZE,))
        shared_dense1 = Dense(256, activation='relu')(input_layer)
        shared_dense2 = Dense(128, activation='relu')(shared_dense1)

        tower_branch = Dense(NUM_POS * 6, activation='linear', name='place_tower')(shared_dense2)
        upgrade_branch = Dense(NUM_POS * 2, activation='linear', name='upgrade_tower')(shared_dense2)
        do_nothing = Dense(1, activation='linear', name='do_nothing')(shared_dense2)

        output_layer = Concatenate()([tower_branch, upgrade_branch, do_nothing])
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])

    return model

model = init()


class Agent:
    def __init__(self, epsilon):
        self.total_rewards = []
        try:
            with open('memory.pkl', 'rb') as f:
                self.memory = pickle.load(f)
        except FileNotFoundError:
            # If the file does not exist, initialize an empty deque with maxlen=5000
            self.memory = deque([], maxlen=5000)        
        self.gamma = 0.98 #discount
        self.epsilon = epsilon 
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.model = model
        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def save_total_rewards(self, filename="total_rewards.txt"):
        with open(filename, "w") as file:
            for reward in self.total_rewards:
                file.write(str(reward) + "\n")

    def update_total_rewards(self, episode_reward):
        self.total_rewards.append(episode_reward)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done): #equivalent of update_replay_memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            print("Performing random action")
            while True:
                random_action = random.randrange(ACTION_SIZE)
                if Game.valid_action(random_action):
                    print(f"Valid random action {action_conversion(random_action)}, number {random_action}")
                    return random_action
                else:
                    continue
        
        print("Performing selected action")
        state = np.array(state)
        state = np.expand_dims(state, axis=0)  # add a batch dimension

        act_values = self.model.predict(state) # action
        act_values = act_values.flatten()
        print(act_values)
        sorted_qs = np.argsort(act_values)[::-1] # sort q values in descending order

        # go through and pick the highest valid q-value

        for action_idx in sorted_qs:
            if Game.valid_action(action_idx):
                print(f"Valid selected action {action_conversion(action_idx)}")
                return action_idx
            else:
                print(f"Invalid selected action {action_conversion(action_idx)}")
    
    #DQN main model/target model training largely from here:
    #https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/

    def replay(self, batch_size): # training
        minibatch = random.sample(self.memory, batch_size)

        #experience = (state, action, reward, next_state, done)

        current_states = np.array([experience[0] for experience in minibatch])
        current_qs_list = self.model.predict(current_states)

        next_states = np.array([experience[3] for experience in minibatch])
        future_qs_list = self.target_model.predict(next_states)
        x = []
        y = []
        for index, (state, action, reward, next_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + (self.gamma * max_future_q) 
            else:
                new_q = reward
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            x.append(state)
            y.append(current_qs)
        x = np.array(x)
        y = np.array(y)
        print(f"Q-values: {y}")
        self.model.fit(x, y, epochs=1, verbose=0,)

    
    def save(self):
        print("Saving model")
        self.model.save("models/model.keras")
    
    # not necessary function just here incase
    def load(self):
        if os.path.exists("models/model.keras"):
            self.model = load_model("models/model.keras")
            print("Previous model loaded")
        else:
            print("No model??? How")

def action_conversion(action):
    if action <= 443:
        pos = action // 6
        index = action % 6
        return f"Place tower at position {pos} and type {index}"
    elif action >= 444 and action < 592:
        pos = (action-444) // 2
        upgrade = action % 2
        return f"Upgrade tower at position {pos} and type {upgrade}"
    else:
        return "Do nothing"

            

if __name__ == '__main__':
    
    agent = Agent(0.53) #control epsilon here lol
    print(agent.model.summary())
    #keras.utils.plot_model(agent.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    #initialize game
    Game = Browser()
    Game.launch()
    Game.start_game()

    for episode in range(100):
        done = False
        Game.reset_game()
        total_reward = 0
        steps = 0
        round = 0
        while not done:
            while True:
                # sometimes this can return none, force it to not
                state = Game.get_game_state()
                #print(state)
                if state is not None:
                    break        
            action = agent.act(state)
            #print(f"we want to take action {action}")
            next_state, reward, done = Game.step(action) # next state is the state 5 seconds later, this is GIVEN already by game.step
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            round = state[2]
            state = next_state # don't think this line does anything
            steps += 1
            if steps % 16 == 0:
                if len(agent.memory) > 32:
                    print("memory")
                    agent.replay(32)
            time.sleep(1)
        if episode % 4 == 0:
            agent.update_target_model()
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            with open('epsilon_values.txt', 'a') as file:
                file.write(f"{agent.epsilon}\n")

        agent.update_total_rewards(total_reward) # can ignore

        with open('data.txt', 'a') as file:
            file.write(f"Episode {episode}: Total Reward = {total_reward}, Steps = {steps}, Epsilon = {agent.epsilon}, Round = {round}\n")
        with open('memory.pkl', 'wb') as f: # dump replay memory
            pickle.dump(agent.memory, f)
        agent.save()
    
    agent.save_total_rewards()


    agent.save()