import json 
import glob
import os
from collections import defaultdict
#['button-press-topdown-v2', 'button-press-v2', 'door-lock-v2', 'door-unlock-v2', 'door-open-v2', 'door-close-v2', 'drawer-open-v2', 'drawer-close-v2', 'window-open-v2', 'window-close-v2', 'faucet-open-v2', 'faucet-close-v2', 'handle-press-v2', 'coffee-button-v2'])

target_list = ['button_press_topdown','button_press','door_lock','door_pull','drawer','window_horizontal','faucet','handle_press','coffee']
files_dir = 'metaworld/all_envs/all_envs'
save_dir  = 'metaworld/all_envs'



for task_name in target_list:
    json_file = json.load(open(os.path.join(files_dir,task_name+'.json')))
    ret_json = defaultdict(list)
    for pos ,list_ in json_file.items():
        for variant in list_:
            ret = all(ele in target_list for ele in variant)
            if ret:
                ret_json[pos].append(variant)

    json.dump(ret_json,open(os.path.join(save_dir,task_name+'.json'),'w'))
    