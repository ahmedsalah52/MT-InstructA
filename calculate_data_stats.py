import json 

def calculate_data_stats(data_dir):
    table=[]
    total_success_rate = 0
    for task_name , episodes in data_dir.items():
        task_success = 0
        for episode in episodes:
            task_success += episode[-1]['success']
        
        task_success_rate = float(task_success) / len(episodes)
        total_success_rate += task_success_rate
        print(task_name , task_success_rate)
        table.append([task_name,task_success_rate])
    table.append(['total_success_rate',total_success_rate/len(data_dir.items())])
    return table


data_dict_dir = '/system/user/publicdata/mansour_datasets/metaworld/generated_data/dataset_dict.json'


data_dir = json.load(open(data_dict_dir))

print(calculate_data_stats(data_dir))
"""
button-press-topdown-v2 0.8198757763975155                                                                                                  
button-press-v2         0.9201680672268907                                                                                                          
door-lock-v2            1.0                                                                                                                            
door-unlock-v2          0.8029197080291971                                                                                                           
door-open-v2            0.954248366013072                                                                                                              
door-close-v2           0.9378238341968912                                                                                                            
drawer-open-v2          0.9955357142857143                                                                                                           
drawer-close-v2         0.9903225806451613                                                                                                          
window-open-v2          0.970873786407767                                                                                                            
window-close-v2         1.0                                                                                                                         
faucet-open-v2          1.0                                                                                                                          
faucet-close-v2         0.9634146341463414                                                                                                          
handle-press-v2         0.9933110367892977                                                                                                          
coffee-button-v2        0.722972972972973 

total_success_rate      0.9336761769364872       


"""