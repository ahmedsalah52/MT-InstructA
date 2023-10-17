import json 
import os
import sys


def main():
    from_dir = sys.argv[1]
    to_dir = sys.argv[2]
    
    with open(from_dir) as f:
        from_dict = json.load(f)
    
    with open(to_dir) as f:
        to_dict = json.load(f)

    for task in from_dict:
        print('task to edit ', task)
        to_dict[task] = from_dict[task]

    #save json to to

    with open(to_dir,'w') as f:
        json.dump(to_dict,f)
    