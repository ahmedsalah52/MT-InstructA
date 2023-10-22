import torch
from train_utils.model import TL_model
from train_utils.args import  parser 

from meta_env import meta_env,task_manager

import cv2
from PIL import Image
import numpy as np
import random
import os
import shutil
class video():
    def __init__(self,save_dir,save_video,res=(1920,1080)):
        self.save_dir = save_dir
        self.save_video = save_video
        self.res = res
        self.video = []
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        
    def imshow_obs(self,env,instruction=None):
        behindGripper  = env.render(offscreen= True,camera_name='behindGripper')
        topview        = env.render(offscreen= True,camera_name='topview')
        topview        = cv2.rotate(topview, cv2.ROTATE_180)
        behindGripper  = cv2.rotate(behindGripper, cv2.ROTATE_180)

        conc_image  = cv2.hconcat([behindGripper,topview])
        white_border = np.zeros((40,conc_image.shape[1],3),dtype=np.uint8)
        white_border.fill(255)
        conc_image = cv2.vconcat([conc_image,white_border])
        if instruction is not None:
            cv2.putText(conc_image, instruction, (5,conc_image.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        final_frame = cv2.resize(cv2.cvtColor( conc_image, cv2.COLOR_RGB2BGR),self.res)
        if self.save_video:
            self.video.append(final_frame)
        cv2.imshow('image',final_frame)
        return cv2.waitKey(0)
    
    def write_video(self,task_name):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(self.save_dir,task_name+'.mp4'), fourcc, 30.0, (self.res[0],self.res[1]))
        for i in self.video:
            out.write(i)
        out.release()
        
        


def get_visual_obs(env):
    corner         = env.render(offscreen= True,camera_name='corner') # corner,2,3, corner2, topview, gripperPOV, behindGripper'
    corner2        = env.render(offscreen= True,camera_name='corner2')
    behindGripper  = env.render(offscreen= True,camera_name='behindGripper')
    corner3        = env.render(offscreen= True,camera_name='corner3')
    topview        = env.render(offscreen= True,camera_name='topview')
    
    images = [cv2.resize(corner,(224,224)),       
            cv2.resize(corner2,(224,224)),      
            cv2.resize(behindGripper,(224,224)),
            cv2.resize(corner3,(224,224)),      
            cv2.resize(topview,(224,224))      
    ]

    return np.array(images)
def main():
    args = parser.parse_args()
    print('save video ',args.save_video)
    video_man = video(save_dir=args.video_dir,save_video=args.save_video,res=args.video_res)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TL_model(args=args,tasks_commands=None,env=None,wandb_logger=None,seed=None).to(device)
    model = TL_model.load_from_checkpoint(args.load_checkpoint_path,args=args,tasks_commands=None,env=None,wandb_logger=None,seed=args.seed)
    model.eval()
    taskname =  random.choice(args.tasks)
    pos = 2
    multi = True
    variant = None #['faucet','window_horizontal','coffee']

    task_man = task_manager(taskname,pos=pos,variant=variant,multi=multi,general_model=True)


    env = task_man.reset()
   
    obs = env.reset()  # Reset environment
    key = video_man.imshow_obs(env)
    instruction = input('enter the instruction:')
    while 1:
        step_input = {'instruction':[instruction]}
        images =  [model.model.preprocess_image(Image.fromarray(np.uint8(img))) for img in get_visual_obs(env)]
        step_input['images']   = torch.stack(images).unsqueeze(0).to(model.device)
        step_input['hand_pos'] = torch.tensor(np.concatenate((obs[0:4],obs[18:22]),axis =0)).to(torch.float32).unsqueeze(0).to(model.device)
        a = model.model(step_input)
        obs, reward, done, info = env.step(a.detach().cpu().numpy()[0]) 
        print(instruction , reward)
        #print(np.concatenate((obs[0:4],obs[18:22]),axis =0))
        key = video_man.imshow_obs(env,instruction)
        
        if key & 0xFF == ord('m'):
            instruction = input('enter the instruction:')
        if key & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    env.close()
    video_man.write_video(args.video_exp_name)

if __name__ == "__main__":
    main()
    