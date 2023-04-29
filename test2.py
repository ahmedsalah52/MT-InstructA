import metaworld
import random
import cv2
ml10 = metaworld.ML10() # Construct the benchmark, sampling tasks

training_envs = []
for name, env_cls in ml10.train_classes.items():
    env = env_cls()
    task = random.choice([task for task in ml10.train_tasks
                            if task.env_name == name])
    env.set_task(task)
    training_envs.append(env)

for env in training_envs:
    obs = env.reset()  # Reset environment
    a = env.action_space.sample()  # Sample an action
    obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
    corner         = env.render(offscreen= True,camera_name='corner')# corner,2,3, corner2, topview, gripperPOV, behindGripper'
    corner2        = env.render(offscreen= True,camera_name='corner2')
    behindGripper  = env.render(offscreen= True,camera_name='behindGripper')
    corner3        = env.render(offscreen= True,camera_name='corner3')
    topview        = env.render(offscreen= True,camera_name='topview')

    all     = cv2.hconcat([corner,corner2,corner3,cv2.flip(behindGripper, 0),topview])

    cv2.imshow('show',cv2.cvtColor(all, cv2.COLOR_RGB2BGR))
   
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    
    cv2.destroyAllWindows()

    env.close()