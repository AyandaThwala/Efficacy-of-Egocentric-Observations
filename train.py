import os
import gym
import time
import random

from library import *
from envs.envs import *
from envs.wrappers import *

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
            path='models',
            load_model=False,
            save_model=True,
            max_episodes=50000,
            learning_starts=int(1e4),
            replay_buffer_size=int(1e5),
            train_freq=4,
            target_update_freq=int(1e3),
            batch_size=32,
            gamma=0.95,
            learning_rate=1e-4,
            eps_initial=1.0,
            eps_final=0.1,
            eps_success=0.96,
            timesteps_success = 100,
            mean_episodes=100,
            eps_timesteps=int(5e5),
            print_freq=10):
    
    ### Initialising
    start_time = time.time()
    eps_schedule = LinearSchedule(eps_timesteps, eps_final, eps_initial)
    replay_buffer = ReplayBuffer(replay_buffer_size, batch_size)
    
    agent = Agent(env, gamma=gamma, learning_rate=learning_rate, replay_buffer=replay_buffer, path=path)
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

    times = []
    timesteps = []

    evaluate_returns = []
    evaluate_successes = []

    avg_returns = []
    avg_successes = []

    avg_return = 0
    success_rate = 0
    avg_return_best = 0
    success_rate_best = 0
    steps = 0

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

            np.savetxt('episode_returns.csv', episode_returns, delimiter=',', fmt='%1.3f')
            np.savetxt('episode_successes.csv', episode_successes, delimiter=',', fmt='%1.3f')

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

            np.savetxt('evaluate_returns.csv', evaluate_returns, delimiter=',', fmt='%1.3f')
            np.savetxt('evaluate_successes.csv', evaluate_successes, delimiter=',', fmt='%1.3f')
            
            if success_rate > success_rate_best:
                avg_return_best = avg_return
                success_rate_best = success_rate

                if save_model:
                    ### Save models
                    agent.save()
                    print("\nModels saved. ar: {}, sr: {}\n".format(avg_return_best, success_rate_best))
            
            if success_rate_best > eps_success:
                print("\nTask solved: success_rate > {}\n".format(eps_success))  
                break
                
        ### Print training progress
        if done and print_freq is not None and episode % print_freq == 0:
            avg_return_ = np.mean(episode_returns[-mean_episodes-1:-1])
            success_rate_ = np.mean(episode_successes[-mean_episodes-1:-1]) 

            avg_returns.append(avg_return_)
            avg_successes.append(success_rate_)

            np.savetxt('avg_return_.csv', avg_returns, delimiter=',', fmt='%1.3f')
            np.savetxt('success_rate_.csv', avg_successes, delimiter=',', fmt='%1.3f')

            print("--------------------------------------------------------")
            print("steps {}".format(steps))
            print("episodes {}".format(episode))
            print("mission {}".format(obs['mission']))
            print("average return: current {}, eval_current {}, eval_best {}".format(avg_return_,avg_return,avg_return_best))
            print("success rate: current {}, eval_current {}, eval_best {}".format(success_rate_,success_rate,success_rate_best))
            print("% time spent exploring {}".format(int(100 * eps_schedule(steps))))
            print("--- %s seconds ---" % (time.time() - start_time)) 
            print("--------------------------------------------------------")

            times.append(time.time() - start_time)
            timesteps.append(steps)

            np.savetxt('time.csv', times, delimiter=',', fmt='%1.3f')
            np.savetxt('timestep.csv', timesteps, delimiter=',', fmt='%1.3f')   


    print("--- %s seconds ---" % (time.time() - start_time)) 
    times.append(time.time() - start_time)
    timesteps.append(steps)
    np.savetxt('time.csv', times, delimiter=',', fmt='%1.3f')
    np.savetxt('timestep.csv', timesteps, delimiter=',', fmt='%1.3f')   
    agent.save()

    return agent, model, episode_returns, episode_successes


if __name__ == '__main__':    
    env_key="MiniGrid-Empty-16x16-v0"
    env = gym.make(env_key)
    env = FullyObsWrapper(env, egocentric=True)
    env = RGBImgObsWrapper(env) 
    path='models/{}'.format(env_key)

    train(env, path = path, save_model=True, load_model=False)