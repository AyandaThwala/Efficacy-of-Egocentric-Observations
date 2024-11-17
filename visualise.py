import torch
import imageio
import numpy as np

from library import *
from wrappers import *

from matplotlib import pyplot as plt

def visualise(egocentric, domain, env_key, max_episodes=2, max_trajectory=50):

    # Set these for environment and characteristic being evaluated
    d="h"                               #       t=test  e=easy  m=middle_child  h=hard
    c=domain                            #       characteristic being tested:    Obstacle(avoidance), Navi(gation), Task(completion)
    env_key=env_key                     #       https://github.com/rohitrango/gym-minigrid          (choose your favourite environment)
    max_episodes = max_episodes
    max_trajectory = max_trajectory
    x = (1, 3)  

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

            print('Saving video ...')    
            images = []

            for episode in range(max_episodes):
                obs = env.reset()
                mission = tokenize(obs['mission'], model['vocab'])

                
                
                done = False
                for _ in range(max_trajectory):

                    image_allocentric = env.render("rgb_array", highlight=False)
                    image_egocentric = obs["image"]

                    fig, axs = plt.subplots(1, 2)
                    axs = axs.flatten()

                    axs[0].set_title("ref: Allocentric", fontsize=20)
                    axs[0].set_xticks([])
                    axs[0].set_yticks([])
                    axs[0].imshow(image_allocentric)

                    axs[1].set_title(name, fontsize=20)
                    axs[1].set_xticks([])
                    axs[1].set_yticks([])
                    axs[1].imshow(image_egocentric)

                    fig.suptitle("Mission: "+obs["mission"], fontsize=20)
                    fig.tight_layout()
                    fig.subplots_adjust(top=1)
                    images.append(fig_image(fig))

                    plt.close(fig)

                    action = select_action(model,obs['image'],mission)
                    obs, _, done, _ = env.step(action)

                    if done:
                        print(f"Episode: {episode}, Mission success.\n")
                        break
                if not done:   
                    print(f"Episode: {episode}, Mission failed.\n")
        
            imageio.mimsave(f"images/trained_{name}_{n}.gif",images,fps=5)

def fig_image(fig):
    fig.gca().margins(0)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


if __name__ == '__main__':

    # visualise(egocentric=True, domain="Navi", env_key="MiniGrid-SimpleCrossingS9N3-v0")
    # visualise(egocentric=False, domain="Navi", env_key="MiniGrid-SimpleCrossingS9N3-v0")

    visualise(egocentric=True, domain="Obstacle", env_key="MiniGrid-Dynamic-Obstacles-8x8-v0")
    visualise(egocentric=False, domain="Obstacle", env_key="MiniGrid-Dynamic-Obstacles-8x8-v0")

    visualise(egocentric=True, domain="Task", env_key="MiniGrid-Fetch-8x8-N3-v0")
    visualise(egocentric=False, domain="Task", env_key="MiniGrid-Fetch-8x8-N3-v0")