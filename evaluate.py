import torch
import imageio
import numpy as np

from library import *
from wrappers import *

from matplotlib import pyplot as plt

# Set these for the experiment you want to conduct
n=1                                                        #       run number
d="t"                                                       #       t=test  e=easy  m=middle_child  h=hard
c="Navi"                                                #       characteristic being tested:    Obstacle(avoidance), Navi(gation), Task(completion)
egocentric = False
env_key="MiniGrid-LavaCrossingS9N3-v0"                 #       https://github.com/rohitrango/gym-minigrid          (choose your favourite environment)

if egocentric:
    name = "Egocentric"
    name=f"{name}_{d}{c}_{n}"
else:
    name = "Allocentric"
    name=f"{name}_{d}Obstacle_{n}"

env = gym.make(env_key) 
env = FullyObsWrapper(env, egocentric=egocentric) 
env = RGBImgObsWrapper(env)
path=f'models/{name}'

def fig_image(fig):
    fig.gca().margins(0)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


if __name__ == '__main__':
    print('Loading ...')
    model = load(path, env)

    print('Saving video ...')    
    images = []

    max_episodes = 4
    max_trajectory = 20

    with torch.no_grad():
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
                action = select_action(model,obs['image'],mission)
                obs, reward, done, _ = env.step(action)

                plt.close(fig)

                if done:
                    break
                
            if done:
                print("Mission success.\n")
            else:
                print("Mission failed.\n")
          
    imageio.mimsave(f"images/trained_{name}.gif",images,fps=10)