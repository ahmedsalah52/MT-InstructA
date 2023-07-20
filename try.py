import threading
from  tqdm import tqdm
def get_episode(num,num2,ret):
    ret.append(num+num2)


episodes = 10
data_dict = {}
taskname = 'button-press-topdown-v2'
threads = []
rets = []
args = [1,2,rets]
for i in range(episodes):
    t = threading.Thread(target=get_episode, args=(args))  
    t.start()
    threads.append(t)

for t in tqdm(threads):
    t.join()
    
for episode_num in range(episodes):
    episode_dict = rets[episode_num]
    data_dict[episode_num] = {'task_name':taskname,'data':episode_dict}

print(data_dict)
