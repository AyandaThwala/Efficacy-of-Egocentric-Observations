import torch
import imageio
import argparse
import numpy as np

from library import *
from envs.envs import *
from envs.wrappers import *

from matplotlib import pyplot as plt
from gym_minigrid.window import Window

egocentric = True

if egocentric:
    modality = "Egocentric"
else:
    modality = "Allocentric"

parser = argparse.ArgumentParser()

parser.add_argument(
    '--env_key',
    default="MiniGrid-Empty-16x16-v0",
    help="Environment"
)

parser.add_argument(
    '--exp',
    default=None,
    help="Task expression"
)

parser.add_argument(
    '--num_dists',
    type=int,
    default=1,
    help="Number of distractors"
)

parser.add_argument(
    '--size',
    type=int,
    default=7,
    help="Grid size"
)

parser.add_argument(
    '--save',
    default=True,
    help="draw what the agent sees",
    action='store_true'
)

args = parser.parse_args()

env = gym.make(args.env_key)

env = FullyObsWrapper(env, egocentric=egocentric)
env = RGBImgObsWrapper(env) 


def fig_image(fig):
    fig.gca().margins(0)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image

if __name__ == '__main__':
    path='models/{}'.format(args.env_key)
      
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

                axs[0].set_title("ref: Allocentric obs", fontsize=20)
                axs[0].set_xticks([])
                axs[0].set_yticks([])
                axs[0].imshow(image_allocentric)

                axs[1].set_title(f"modality: {modality}", fontsize=20)
                axs[1].set_xticks([])
                axs[1].set_yticks([])
                axs[1].imshow(image_egocentric)

                fig.suptitle("Mission: "+obs["mission"], fontsize=20)
                fig.tight_layout()
                fig.subplots_adjust(top=1)
                images.append(fig_image(fig))

                plt.close(fig)

                action = select_action(model,obs['image'],mission)
                obs, reward, done, _ = env.step(action)
                
                if done:
                    break

            if done:
                print("Mission success.\n")
            else:
                print("Mission failed.\n")
            
    if args.save:
        imageio.mimsave("images/trained_agent_{}.gif".format(args.env_key),images,fps=0.5)

