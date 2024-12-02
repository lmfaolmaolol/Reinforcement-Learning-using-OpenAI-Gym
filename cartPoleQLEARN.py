import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):

    env = gym.make('CartPole-v1', render_mode='human' if render else None)

    # Divide position and velocity into segments(Discretization)
    pos_cart = np.linspace(-4.8,4.8, 40)    # Between -1.2 and 0.6
    vel_cart = np.linspace(-3, 3, 40) 
    pole_angle=np.linspace(-0.418,0.418,40)
    pole_angleVel = np.linspace(-10, 10, 40)  # Between -0.07 and 0.07

    if(is_training):
        q = np.zeros((len(pos_cart), len(vel_cart),len(pole_angle),len(pole_angleVel), env.action_space.n)) # init a 20x20x3 array
    else:
        f = open('cartPole.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.2 # alpha or learning rate(relies heavily on previously explored data so it focuses more on exploitation later on in the algo)
    discount_factor_g = 0.99 # gamma or discount factor(high as we need it to focu on long term goals)

    epsilon = 1         # 1 = 100% random actions
    decay=2/episodes
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]      
        state_p = np.digitize(state[0], pos_cart) - 1
        state_v = np.digitize(state[1], vel_cart) - 1
        state_a = np.digitize(state[2], pole_angle) - 1
        state_av = np.digitize(state[3], pole_angleVel) - 1

        terminated = False          # True when reached goal

        rewards=0

        #epsilon greedy approach
        while(not terminated and rewards>-1000):

            if is_training and rng.random() < epsilon:
                # Choose random action
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v,state_a,state_av, :])

            new_state,reward,terminated,_,_ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_cart)-1
            new_state_v = np.digitize(new_state[1], vel_cart)-1
            new_state_a = np.digitize(new_state[2], pole_angle)-1
            new_state_av = np.digitize(new_state[3], pole_angleVel)-1

            if is_training:
                q[state_p, state_v, state_a,state_av,action] = q[state_p, state_v,state_a,state_av, action] + learning_rate_a * (
                    reward + discount_factor_g*np.max(q[new_state_p, new_state_v,new_state_a,new_state_av,:]) - q[state_p, state_v,state_a,state_av, action]
                )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a=new_state_a
            state_av=new_state_av

            rewards+=reward

        epsilon = max(epsilon-decay,0)

        rewards_per_episode[i] = rewards

        print(rewards)
        print(i)

    env.close()

    # Save Q table to file
    if is_training:
        f = open('cartPole.pkl','wb')
        pickle.dump(q, f)
        f.close()

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(mean_rewards)
    plt.savefig(f'cartPole.png')

if __name__ == '__main__':
    run(1, is_training=False, render=True)
