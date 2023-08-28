import json,glob,os
line = "sawyer_button_press_topdown-handle_press-basketball,0.35,-0.35,0.xml"
from collections import defaultdict

def get_sorted_from(line):
    line = line[7:-4]
    print(line)
    envs,l,m,r = line.split(',')

    envs = envs.split('-')
    poses = [float(l),float(m),float(r)]

    sorted_envs = [x for _, x in sorted(zip(poses,envs))]

    return sorted_envs


def get_tasks_list(main_file):
    env_txt_file = open(main_file,'r')
    print('main',main_file)
    main_env_name = main_file.split('.')[0].split('/')[-1].split('-')[0][7:]
    env_txt_lines = env_txt_file.read().split('\n')

    out = defaultdict(list)
    for line in env_txt_lines:
        sorted_line  = get_sorted_from(line)
        print(sorted_line)
        idx = sorted_line.index(main_env_name)
        out[idx].append(sorted_line)
    return out


def change_files(in_dir,save_dir):
    
    in_files = glob.glob(os.path.join(in_dir,'*'))
    for file in in_files:
        tasks_list = get_tasks_list(file)
        new_file_name = file.split('/')[-1].split('.')[0]+'.json'
        new_file_name = new_file_name[7:]
        
        with open(os.path.join(save_dir,new_file_name), 'w') as f:
            json.dump(tasks_list, f)

                
change_files('metaworld/all_envs_old','metaworld/all_envs')