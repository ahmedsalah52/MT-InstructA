import torch
from train_utils.model import base_model
from train_utils.args import  parser 

from meta_env import meta_env
import cv2
from PIL import Image
import numpy as np
def main():
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = base_model(args=args,tasks_commands=None,env=None,wandb_logger=None,seed=None).to(device)
    model = base_model.load_from_checkpoint(args.load_checkpoint_path,args=args,tasks_commands=None,env=None,wandb_logger=None,seed=args.seed)
    model.eval()
    task = 'faucet-open-v2'
    pos = 2
    env = meta_env(task,pos,save_images=True,wandb_render = False,wandb_log = False,general_model=True)
    #print('task variant ', env.env.current_task_variant)
    obs , info = env.reset()
    cv2.imshow('image',cv2.resize(cv2.cvtColor( env.get_visual_obs_log(), cv2.COLOR_RGB2BGR),(1920,1080)))
    key = cv2.waitKey(0)

    instruction = input('enter the instruction:')

    while 1:
        step_input = {'instruction':[instruction]}
        images = [model.model.preprocess_image(Image.fromarray(np.uint8(img))) for img in info['images']]
        step_input['images']   = torch.stack(images).unsqueeze(0).to(model.device)
        step_input['hand_pos'] = torch.tensor(np.concatenate((obs[0:4],obs[18:22]),axis =0)).to(torch.float32).unsqueeze(0).to(model.device)
        a = model.model(step_input)
        obs, reward, done,success, info = env.step(a.detach().cpu().numpy()[0]) 
        print(instruction , reward)
        cv2.imshow('image',cv2.resize(cv2.cvtColor( env.get_visual_obs_log(), cv2.COLOR_RGB2BGR),(1920,1080)))
        key = cv2.waitKey(0)
        if key & 0xFF == ord('m'):
            instruction = input('enter the instruction:')
        if key & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    main()
    