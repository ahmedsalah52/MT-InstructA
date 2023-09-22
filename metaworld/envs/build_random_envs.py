import os
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
import mujoco_py
import os, json
import numpy as np
import uuid



class multi_object_man():
    def __init__(self,main_envs_dir = 'metaworld/envs/assets_v2/sawyer_xyz/' ,init_file_name = ''):
        
        self.main_envs_dir = main_envs_dir
        self.init_file_name = init_file_name

        
    def get_new_env(self,main_rot,secondary_files,secondary_poses):
        #poses_list = [0,1,2]
        #dx_idx = poses_list.pop(random.randrange(len(poses_list)))
        
        
        self.file_name = build_env(os.path.join(self.main_envs_dir,self.init_file_name),main_rot,secondary_poses,secondary_files)

        self.text_file_name = self.init_file_name.split('.')[0]+ '.txt'

    
    def get_file_name(self):
        return self.file_name
    
    def multi_env_loaded(self):
        file_path = os.path.join('metaworld/all_envs',self.text_file_name)
        mode = 'r' if os.path.exists(file_path) else 'w'
        
        file = open(file_path, mode) 
        files_list = file.read().split('\n')
        file.close() 

        if self.file_name not in files_list:
            file = open(file_path, 'a') 
            file.write(self.file_name) 
            file.write('\n')
            file.close() 

            file = open('metaworld/all_envs/all_envs.txt', 'a') 
            file.write(self.file_name) 
            file.write('\n')
            file.close() 
        else:
            print('_________________________________________________________ repeated : ',self.file_name)

    def multi_env_not_loaded(self):
        file_path = 'metaworld/all_envs/all_envs_fail.txt'
        mode = 'r' if os.path.exists(file_path) else 'w'

        file = open(file_path,mode) 
        files_list = file.read().split('\n')
        file.close() 

        if self.file_name not in files_list:
            file = open(file_path, 'a') 
            file.write(self.file_name) 
            file.write('\n')
            file.close() 
        else:
            print('_________________________________________________________ repeated failure: ',self.file_name)

def write_mjcf_file(body_trees: list, includes: list, filepath: str):
    root = ET.Element('mujoco')
    for body_tree in body_trees:
        #body_tree.attrib['name'] +='unique'
        for child in body_tree:
            if child.tag == 'joint' or child.tag == 'freejoint':
                body_tree.remove(child)
        root.append(body_tree)
    tree = ET.ElementTree(root)
    with open(filepath, 'wb') as f:
        tree.write(f)


def add_body_to_tree(main_tree, body_file,internal_file_dir,unique):
    # Parse the body file
    body_tree = ET.parse(body_file)
    body_root = body_tree.getroot()

    # Find the body element in the body tree
    body_elem = body_root.find('body')
    
    # Get the name and position of the body
    name = body_elem.attrib['name']+unique
    pos = body_elem.attrib['pos']

    # Create a new body element in the main tree
    new_body_elem = ET.Element('body')
    new_body_elem.attrib['name'] = name
    new_body_elem.attrib['pos'] = '0 0 0'

    # Add the include element to the new body element
    include_elem = ET.Element('include')
    include_elem.attrib['file'] = internal_file_dir
    new_body_elem.append(include_elem)

    # Add the new body element to the main tree
    worldbody_elem = main_tree.find('worldbody')
    worldbody_elem.append(new_body_elem)

    return main_tree
def get_tree_includes(tree):
    root = tree.getroot()
   
    includes = root.findall('include')

    return includes



def get_bodies(tree):
    root = tree.getroot()
    #worldbody = root.find('worldbody')
    #bodies = worldbody.findall('body') 
    bodies = root.findall(".//body")
    return bodies

def kill_duplicates(includes):
    check = []
    ret = []
    for inc in includes:
        if inc.attrib['file'] not in check:
            ret.append(inc)
            check.append(inc.attrib['file'])
    return ret
            
def edit_body_pos(body_list, pos_offset_rot):
    pos_offset = pos_offset_rot[0:2]
    new_euler  = pos_offset_rot[2:]
    bodies = []
    for body in body_list:
        body.set('name', body.attrib.get('name')+str(pos_offset))
        
        pos = body.get('pos').split()
        new_pos = [ pos_offset[0], float(pos[1])+pos_offset[1], float(pos[2])]
        body.set('pos', ' '.join(map(str, new_pos)))

        # Modify the rotation
        euler = body.get('euler')
        if euler != None:
            euler = euler.split()
            new_euler = [float(euler[0])+new_euler[0] , float(euler[1])+new_euler[1], float(euler[2])+new_euler[2] ]
        body.set('euler', ' '.join(map(str, new_euler)))

        bodies.append(body)
    return bodies


def add_includes_to_tree(tree, includes):
    root = tree.getroot()
    # Get a list of existing include elements in the xml file
    existing_includes = root.findall('include')

    # Remove any existing includes that match the ones we're trying to add
    for include in includes:
        for existing_include in existing_includes:
            if include.attrib == existing_include.attrib:
                root.remove(existing_include)

    # Add the new includes to the xml file
    for include in includes:
        root.insert(0, include)

    return tree

def build_env(main_env_path,main_rot,sec_poses,sec_files):
    #load main env xml file
    #load to random envs files
    #pick the body and independents sections in a dict
    #edit the body x position of both games
    #add the two games, independents to the main file,
    # save the file to the same dir
    #return the file name
    
    main_env_name = str(uuid.uuid4())

    out_file_name = ''
    bodies_names = []
    secondary_envs_names = []
    multi_envs_dir = 'metaworld/envs/assets_v2/sawyer_xyz_multi/'
    xml_dir   = os.path.split(main_env_path)[0]
    #main_env_name = os.path.split(main_env_path)[1].split('.')[0]
    main_tree = ET.parse(main_env_path)
    root = main_tree.getroot()
    bodies = root.findall(".//body")
    for body in bodies:
        euler = body.get('euler')
        if euler != None:
            euler = euler.split()
            euler = body.get('euler').split()
            main_rot = [float(euler[0])+main_rot[0] , float(euler[1])+main_rot[1], float(euler[2])+main_rot[2] ]
        body.set('euler', ' '.join(map(str, main_rot)))

    mjcfs_save_dir = os.path.join(multi_envs_dir,'mjcfs',main_env_name)
    if not os.path.isdir(mjcfs_save_dir):
        os.system('mkdir -p '+mjcfs_save_dir)

    mjcf_dir       = os.path.join('../sawyer_xyz_multi/mjcfs',main_env_name)

  
    secondary_envs_trees = []
    for file in sec_files:
        secondary_envs_trees.append(ET.parse(os.path.join(xml_dir,file)))
        secondary_env_name = file.split('.')[0][7:]
        secondary_envs_names.append(secondary_env_name)
        bodies_names.append(secondary_env_name+'.mjcf')
    
    envs = defaultdict(list)
    for i,(pos_offset , tree) in enumerate(zip(sec_poses,secondary_envs_trees)):
        
        bodies_names[i] = bodies_names[i]
        bodies = get_bodies(tree)
        write_mjcf_file(edit_body_pos(bodies,pos_offset), get_tree_includes(tree)  , os.path.join(mjcfs_save_dir,bodies_names[i]))
        envs['includes'] += get_tree_includes(tree)        

    envs['includes'] = kill_duplicates(envs['includes'])
    main_tree = add_includes_to_tree(main_tree,  envs['includes'])
    
    for i , body_name in enumerate(bodies_names):
        main_tree = add_body_to_tree(main_tree,  os.path.join(mjcfs_save_dir,body_name),os.path.join(mjcf_dir,body_name),str(i))



    # Create the new element
    new_element = ET.Element('size')
    new_element.set('njmax', '8000')
    new_element.set('nconmax', '4000')

    # Find the appropriate location in the main_tree where you want to add the new element
    # For example, if you want to add it as a child of the root element, you can do:
    root = main_tree.getroot()
    root.append(new_element)


    """  out_file_name += main_env_name
    for sec_env in secondary_envs_names:
        out_file_name += '-'
        out_file_name += sec_env
    """
    
    """for pos in [main_pos] + poses:
        out_file_name += ','
        out_file_name += str(pos)"""

    out_file_name = main_env_name + '.xml'

    main_tree.write(os.path.join(multi_envs_dir,out_file_name))
    return out_file_name
    




class Multi_task_env():
    def __init__(self):
        self.main_poses_dict  = json.load(open(os.path.join('configs/env_configs/main_poses_dict.json')))
        self.json_file_data   = None

    def generate_env(self,main_file,main_pos_index,task_variant):
        main_poses_dict = self.main_poses_dict 
        poses_list    = [0,1,2]

        main_task_name = main_file.split('.')[0]

        main_envs_dir = 'metaworld/envs/assets_v2/sawyer_xyz/'
       


        main_task_name = main_task_name[7:] # remove sawyer_
        if self.json_file_data == None: 
            self.json_file_data = json.load(open(os.path.join('metaworld/all_envs',main_task_name+'.json')))
        if main_pos_index == None or main_pos_index >= 3: main_pos_index = random.choice(poses_list)
        if task_variant == None:
            task_variants    = self.json_file_data[str(main_pos_index)]
            self.file_order  = random.choice(range(len(task_variants)))
            task_variant     = task_variants[self.file_order][:]
            
        self.current_task_variant = task_variant
        
        main_task_index = task_variant.index(main_task_name)
        task_variant.pop(main_task_index)
        poses_list.pop(main_task_index)
        
        task_key = main_task_name if main_task_name in main_poses_dict.keys() else 'default' 
        main_task_offsets = main_poses_dict[task_key]['main'][str(main_task_index)]['offset']
        main_task_range   = main_poses_dict[task_key]['main'][str(main_task_index)]['range']
        main_rot          = main_poses_dict[task_key]['main'][str(main_task_index)]['rot']
      
        

        secondary_poses= []  
        for i in range(len(poses_list)):
            task_key = task_variant[i] if task_variant[i] in main_poses_dict.keys() else 'default' 
            task_offsets = main_poses_dict[task_key]['sec'][str(poses_list[i])]['offset']
            task_range   = main_poses_dict[task_key]['sec'][str(poses_list[i])]['range']
            sec_rot      = main_poses_dict[task_key]['sec'][str(poses_list[i])]['rot']
            task_offsets_min = np.array(task_offsets) - np.array(task_range)
            task_offsets_max = np.array(task_offsets) + np.array(task_range)
            
            sec_offset = list(np.random.uniform(task_offsets_min, task_offsets_max))
            secondary_poses.append(sec_offset+sec_rot)
            
        task_variant = ['sawyer_'+task+'.xml' for task in task_variant]
        
        self.file_name = build_env(os.path.join(main_envs_dir ,main_file),main_rot,secondary_poses,task_variant)
        self.task_offsets_min = np.array(main_task_offsets) - np.array(main_task_range)
        self.task_offsets_max = np.array(main_task_offsets) + np.array(main_task_range)
        min_x = self.task_offsets_min[0]
        max_x = self.task_offsets_max[0]
        self.hand_init_pos_  = [np.random.uniform(min(-0.1,min_x) ,max(0.1,max_x)),np.random.uniform(0.4,0.7), np.random.uniform(0.15,0.3)]
        self.main_pos_index = main_pos_index
        