import json 




data_dict_dir = '/system/user/publicdata/mansour_datasets/metaworld/generated_data/dataset_dict.json'


data_dir = json.load(open(data_dict_dir))
total_success_rate = 0
for task_name , episodes in data_dir.items():
    task_success = 0
    for episode in episodes:
        task_success += episode[-1]['success']
    
    task_success_rate = float(task_success) / len(episodes)
    total_success_rate += task_success_rate
    print(task_name , task_success_rate)

print('total_success_rate',total_success_rate/len(data_dir.items()))