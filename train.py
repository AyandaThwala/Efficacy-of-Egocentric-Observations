import os
import gym
import time
import wandb
import random
import numpy as np

from library import *
from wrappers import *

def evaluate(args):
    env, model, num_episodes, max_episode_timesteps  = args
    returns, successes = (0, 0)

    for _ in range(num_episodes):
        obs = env.reset()
        mission = tokenize(obs['mission'],model['vocab'])

        for t in range(max_episode_timesteps):
            action = select_action(model, obs['image'], mission)
            new_obs, reward, done, info = env.step(action)
            obs = new_obs
            returns+=reward
            successes+=(reward>0)+0

            if done:
                break

    return [returns, successes]

def train(env,
            path='models/',
            load_model=False,
            save_model=True,
            max_episodes=int(1e6),
            max_timesteps=int(5e2),
            learning_starts=int(1e4),
            replay_buffer_size=int(1e5),
            target_update_freq=int(1e3),
            batch_size=32,
            gamma=0.95,
            learning_rate=1e-4,
            eps_initial=1.0,
            eps_final=0.1,
            eps_success=0.96,
            timesteps_success=200,
            mean_episodes=100,
            eps_timesteps=int(5e5),
            print_freq=10,
            env_n=""):
    
    ### Initialising
    eps_schedule = LinearSchedule(eps_timesteps, eps_final, eps_initial)
    replay_buffer = ReplayBuffer(replay_buffer_size, batch_size)
    
    agent = Agent(env, gamma=gamma, learning_rate=learning_rate, replay_buffer=replay_buffer, path=path)
    agent.save(best=True)

    if load_model and os.path.exists(path):
        model = load(path, env)
        agent.vocab = model['vocab']
        agent.q_func.load_state_dict(model['params'].state_dict())
        agent.target_q_func.load_state_dict(model['params'].state_dict())
        print('RL model loaded')

    model = {'params': agent.q_func, 'vocab': agent.vocab}
    agent.path = path

    # Training  
    episode_returns = []
    episode_successes = []

    avg_returns = []
    avg_successes = []

    evaluate_returns = []
    evaluate_successes = []
    
    avg_return = 0
    success_rate = 0
    avg_return_best = 0
    success_rate_best = 0
    steps = 0
    eval_count = 0

    for episode in range(max_episodes):
        obs = env.reset()
        mission = tokenize(obs['mission'],agent.vocab)

        episode_returns.append(0.0)
        episode_successes.append(0.0)
        done = False
        t = 0

        while not done and t<timesteps_success:
            ### Collecting experience
            if random.random() > eps_schedule(steps):
                action = select_action(model, obs['image'], mission)
            else:
                action =  env.action_space.sample()
            
            new_obs, reward, done, info = env.step(action)

            replay_buffer.add(mission, obs['image'], action, reward, new_obs['image'], done, info)
            obs = new_obs
            episode_returns[-1] += (gamma**t)*reward
            episode_successes[-1] = (t<timesteps_success)*(episode_returns[-1]>0)

            ### Updating agent    
            if steps > learning_starts:
                agent.update_td_loss()

            if steps > learning_starts and steps % target_update_freq == 0:
                agent.update_target_network()

            t += 1    
            steps += 1

        if episode % 500 == 0:
            print("evaluating ...")
            args = [env,
                    model,
                    mean_episodes,
                    timesteps_success]
            returns, successes = evaluate(args)
            avg_return, success_rate = (returns/mean_episodes, successes/mean_episodes)

            evaluate_returns.append(avg_return)
            evaluate_successes.append(success_rate)

            wandb.log({"evaluate return":avg_return,"timestep":steps})

            wandb.log({"evaluate success rate":success_rate,"timestep":steps})
            
            if success_rate > success_rate_best:
                avg_return_best = avg_return
                success_rate_best = success_rate
                
                if save_model:
                    ### Save models
                    agent.save(best=True)
                    print("\nModels saved. ar: {}, sr: {}\n".format(avg_return_best, success_rate_best))
            
            if success_rate_best >= eps_success:
                print("\nTask solved with success_rate: {}\n".format(success_rate_best))  
                eval_count+=1
                
        ### Print training progress
        if print_freq is not None and episode % print_freq == 0:
            
            avg_return_ = round(np.mean(episode_returns[-mean_episodes-1:-1]), 1)
            success_rate_ = np.mean(episode_successes[-mean_episodes-1:-1]) 

            wandb.log({"average return":avg_return_,"timestep":steps})

            wandb.log({"success rate":success_rate_,"timestep":steps})

            wandb.log({"episode":episode,"timestep":steps})

            avg_returns.append(avg_return_)
            avg_successes.append(success_rate_)

            print("--------------------------------------------------------")
            print("steps {}".format(steps))
            print("episodes {}".format(episode))
            print("mission {}".format(obs['mission']))
            print("average return: current {}, eval_current {}, eval_best {}".format(avg_return_,avg_return,avg_return_best))
            print("success rate: current {}, eval_current {}, eval_best {}".format(success_rate_,success_rate,success_rate_best))
            print("% time spent exploring {}".format(int(100 * eps_schedule(steps))))
            print("--------------------------------------------------------")

        if steps >= max_timesteps or eval_count == 10:
            with open(f"{env_n}_steps.txt", "w") as file:
                file.write(f"{steps}")
            with open(f"{env_n}_episodes.txt", "w") as file:
                file.write(f"{episode}")
            break

    np.savetxt(f'{env_n}_avg_return_.csv', avg_returns, delimiter=',', fmt='%1.3f')
    np.savetxt(f'{env_n}_success_rate_.csv', avg_successes, delimiter=',', fmt='%1.3f')

    np.savetxt(f'{env_n}_evaluate_returns.csv', evaluate_returns, delimiter=',', fmt='%1.3f')
    np.savetxt(f'{env_n}_evaluate_successes.csv', evaluate_successes, delimiter=',', fmt='%1.3f')

    agent.save()
    print("model saved")
    
    return agent, model, episode_returns, episode_successes


if __name__ == '__main__':    

    # Set these for the experiment you want to conduct
    n=1                                                    #       run number
    d="h"                                                  #       t=test  e=easy  m=middle_child  h=hard
    c="Navi"                                               #       characteristic being tested:    Obstacle(avoidance), Navi(gation), Task(completion)
    egocentric = True
    env_key="MiniGrid-SimpleCrossingS9N3-v0"                 #       https://github.com/rohitrango/gym-minigrid          (choose your favourite environment)
    
    if egocentric:
        name = "Egocentric"
    else:
        name = "Allocentric"
        
    name=f"{name}_{d}{c}_{n}"
    env = gym.make(env_key) 
    env = FullyObsWrapper(env, egocentric=egocentric) 
    env = RGBImgObsWrapper(env)
    path=f'models/{name}'

    wandb.init(project="Comparative Study", name=name)         #        Uncomment to use weights and biases (and everything that starts with "wandb")

    t0 = time.time()
    train(env, path=path, save_model=True, load_model=False, env_n=name)
    t1 = time.time()

    total = t1-t0
    with open(f"{name}_runtime.txt", "w") as file:
                file.write(f"{total}")