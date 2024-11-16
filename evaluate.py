import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import numpy as np

from library import *
from wrappers import *

from matplotlib import pyplot as plt

max_episodes=200
max_trajectory=50

def evaluate(egocentric, domain, env_key):

    # Set these for environment and characteristic being evaluated
    d="h"                          #       t=test  e=easy  m=middle_child  h=hard
    c=domain                       #       characteristic being tested:    Obstacle(avoidance), Navi(gation), Task(completion)
    env_key=env_key                #       https://github.com/rohitrango/gym-minigrid          (choose your favourite environment)
    step_data = []  
    success_data = []
    x = (1,2)  

    if egocentric:
        modality = "Egocentric"
    else:
        modality = "Allocentric"

    name = f"{modality}_{d}{c}"
    env = gym.make(env_key) 
    env = FullyObsWrapper(env, egocentric=egocentric) 
    env = RGBImgObsWrapper(env)

    with torch.no_grad():
        for n in range(x[0], x[1]):

            print(f"Run: {n}")
            model_name=f"{name}_{n}"
            path=f'models/{model_name}'

            print(f"Loading model: {model_name}")

            print('Loading ...')
            model = load(path, env)

            steps_per_episode = []
            fail_counter = 0

            for episode in range(max_episodes):
                steps = 0
                obs = env.reset()
                mission = tokenize(obs['mission'], model['vocab'])
                
                done = False
                for _ in range(max_trajectory):

                    action = select_action(model,obs['image'],mission)
                    steps += 1
                    obs, reward, done, _ = env.step(action)

                    if done:
                        print(f"Episode: {episode}, Mission success.\n")
                        steps_per_episode.append(steps)
                        break
                if not done:   
                    print(f"Episode: {episode}, Mission failed.\n")
                    fail_counter += 1

            for i in range(fail_counter):
                steps_per_episode.append(np.mean(steps_per_episode))

            success_rate = ((max_episodes - fail_counter) / max_episodes)*100
            success_data.append(success_rate)
            step_data.append(steps_per_episode)   

    plt.boxplot(step_data, labels=[f"Model {i}" for i in range(x[0],x[1])])
    plt.ylabel("Steps")
    plt.title(f'{modality} Domain: {domain}')
    plt.savefig(f'graphs/Evaluation/{modality}_{domain}.png', 
        dpi=300,                
        bbox_inches='tight',   
        transparent=False,      
        pad_inches=0.1         
    )
    # plt.show()
    plt.close()

    return step_data, success_data


if __name__ == '__main__':

    ego_obs_steps, ego_obs_success = evaluate(egocentric=True, domain="Obstacle", env_key="MiniGrid-Dynamic-Obstacles-8x8-v0")

    ello_obs_steps, allo_obs_success = evaluate(egocentric=False, domain="Obstacle", env_key="MiniGrid-Dynamic-Obstacles-8x8-v0")

    ego_nav_steps, ego_nav_success = evaluate(egocentric=True, domain="Navi", env_key="MiniGrid-SimpleCrossingS9N3-v0")

    ello_nav_steps, ello_nav_success = evaluate(egocentric=False, domain="Navi", env_key="MiniGrid-SimpleCrossingS9N3-v0")

    ego_tas_steps, ego_tas_success = evaluate(egocentric=True, domain="Task", env_key="MiniGrid-Fetch-8x8-N3-v0")

    ello_tas_steps, ello_tas_success = evaluate(egocentric=False, domain="Task", env_key="MiniGrid-Fetch-8x8-N3-v0")

    ############################## Best Obstacle Steps ##############################
    bego_obs_steps = ego_obs_steps[np.argmin([np.mean(steps) for steps in ego_obs_steps])]
    ballo_obs_steps = ello_obs_steps[np.argmin([np.mean(steps) for steps in ello_obs_steps])]
    ego_obs_success = np.mean(ego_obs_success)
    allo_obs_success = np.mean(allo_obs_success)

    print(f"ego_obs: {np.argmin([np.mean(steps) for steps in ego_obs_steps])}")
    print(f"allo_obs: {np.argmin([np.mean(steps) for steps in ello_obs_steps])}\n\n")

    ############################## Best Navigation Steps ##############################
    bego_nav_steps = ego_nav_steps[np.argmin([np.mean(steps) for steps in ego_nav_steps])]
    bello_nav_steps = ello_nav_steps[np.argmin([np.mean(steps) for steps in ello_nav_steps])]
    ego_nav_success = np.mean(ego_nav_success)
    ello_nav_success = np.mean(ello_nav_success)

    print(f"ego_nav: {np.argmin([np.mean(steps) for steps in ego_nav_steps])}")
    print(f"allo_nav: {np.argmin([np.mean(steps) for steps in ello_nav_steps])}\n\n")

    ############################## Best Object Steps ##############################
    bego_tas_steps = ego_tas_steps[np.argmin([np.mean(steps) for steps in ego_tas_steps])]
    bello_tas_steps = ello_tas_steps[np.argmin([np.mean(steps) for steps in ello_tas_steps])]
    ego_tas_success = np.mean(ego_tas_success)
    ello_tas_success = np.mean(ello_tas_success)

    print(f"ego_nav: {np.argmin([np.mean(steps) for steps in ego_tas_steps])}")
    print(f"allo_nav: {np.argmin([np.mean(steps) for steps in ello_tas_steps])}\n\n")

    print(f"{len(bego_obs_steps)} {len(ballo_obs_steps)} {len(bego_nav_steps)} {len(bello_nav_steps)} {len(bego_tas_steps)} {len(bello_tas_steps)}")

    print("Success Rates:")
    print(f"Egocentric Obstacle: {ego_obs_success}")
    print(f"Allocentric Obstacle: {allo_obs_success}")
    print(f"Egocentric Navigation: {ego_nav_success}")
    print(f"Allocentric Navigation: {ello_nav_success}")
    print(f"Egocentric Object: {ego_tas_success}")
    print(f"Allocentric Object: {ello_tas_success}")

    # Set the style
    plt.style.use('seaborn')

    # Create figure and subplots with gridspec_kw for height_ratios
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                  gridspec_kw={'height_ratios': [1.2, 1]})

    lenth = 2*max_episodes
    # Compare Egocentric and Allocentric
    data = {
        "Domain": ['Obstacle']*lenth + ['Navi']*lenth + ['Task']*lenth,
        'Steps': np.concatenate([bego_obs_steps, ballo_obs_steps, bego_nav_steps, bello_nav_steps, bego_tas_steps, bello_tas_steps])
    }
    df = pd.DataFrame(data)

    # Create additional grouping data
    sixth = len(df) // 6
    remainder = len(df) % 6

    df['Observation'] = (
        ['Egocentric'] * (sixth + (1 if remainder > 0 else 0)) +  # First sixth + 1 if needed
        ['Allocentric'] * (sixth + (1 if remainder > 1 else 0)) +  # Second sixth + 1 if needed
        ['Egocentric'] * (sixth + (1 if remainder > 2 else 0)) +  # Third sixth + 1 if needed
        ['Allocentric'] * (sixth + (1 if remainder > 3 else 0)) +  # Fourth sixth + 1 if needed
        ['Egocentric'] * (sixth + (1 if remainder > 4 else 0)) +  # Fifth sixth + 1 if needed
        ['Allocentric'] * sixth                                    # Last sixth
    )
    # Plots graph
    sns.boxplot(x="Domain", y='Steps', data=df, hue='Observation', palette='Set2', ax=ax1)

    ax1.set_title('Egocentric vs Allocentric Evaluation', pad=20)
    ax1.set_xlabel('')

    success_data = pd.DataFrame({
        'Domain': ['Obstacle', 'Navigation', 'Task']*2,
        'Observation': ['Egocentric']*3 + ['Allocentric']*3,
        'Success Rate': [ego_obs_success, ego_nav_success, ego_tas_success, allo_obs_success, ello_nav_success, ello_tas_success]
    })

    sns.barplot(x='Domain', y='Success Rate', hue='Observation', data=success_data, palette='Set2', ax=ax2)

    ax2.set_ylim(0, 100)

    for i, bar in enumerate(ax2.patches):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f'{bar.get_height():.1f}%',
            ha='center',
            va='bottom'
        )

    plt.tight_layout()
    plt.savefig(f'graphs/Evaluation/Ego_vs_Allo_evaluation.png', 
        dpi=300,                
        bbox_inches='tight',    
        transparent=False,      
        pad_inches=0.1         
    )

    print("\nSuccess Rate Summary:")
    summary = success_data.groupby('Domain').agg({
        'Success Rate': ['mean', 'std', 'min', 'max']
    }).round(2)
    print(summary)

    egocentric_rates = [ego_obs_success, ego_nav_success, ego_tas_success]
    allocentric_rates = [allo_obs_success, ello_nav_success, ello_tas_success]
    
    diff_by_domain = pd.DataFrame({
        'Domain': ['Obstacle', 'Navigation', 'Task'],
        'Difference (Ego - Allo)': np.array(egocentric_rates) - np.array(allocentric_rates)
    })
    print("\nDifference between approaches (Egocentric - Allocentric):")
    print(diff_by_domain.round(2))

    plt.show()
    plt.close()