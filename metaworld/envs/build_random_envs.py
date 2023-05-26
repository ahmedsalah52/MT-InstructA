import os
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
import mujoco_py

class multi_object_man():
    def __init__(self,delta_x_objects = 0.35 , x_margin = 0.01,main_envs_dir = 'metaworld/envs/assets_v2/sawyer_xyz/' ,init_file_name = ''):
        self.delta_x_objects = delta_x_objects
        self.x_margin = x_margin
        self.main_envs_dir = main_envs_dir
        self.init_file_name = init_file_name

        self.i_iter = 0
        self.j_iter = 0

    def get_new_env(self,secondary_files,dx_idx,poses_list):
        #poses_list = [0,1,2]
        #dx_idx = poses_list.pop(random.randrange(len(poses_list)))
        
        self.dx_dict = {0:-self.delta_x_objects , 1:0 , 2:self.delta_x_objects}
        self.dx = self.dx_dict[dx_idx]
        self.x_margin = self.x_margin

        secondary_poses = [self.dx_dict[poses_list[0]],self.dx_dict[poses_list[1]]]
        
        self.file_name = build_env(os.path.join(self.main_envs_dir,self.init_file_name),secondary_poses,secondary_files,self.dx)

        self.text_file_name = self.init_file_name.split('.')[0]+ '.txt'

    def get_dx_high(self):
        return self.dx + self.x_margin
    
    def get_dx_low(self):
        return self.dx - self.x_margin
    
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

def test_mjcf_file(filepath):
    try:
        # parse the mjcf file
        tree = ET.parse(filepath)
        root = tree.getroot()

        # load the mjcf model
        model = mujoco_py.load_model_from_xml(ET.tostring(root))

        return True
    except Exception as e:
        return False, str(e)

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


def save_mjcf(body_trees, dependency_includes, file_path):
    # Create the root element and add the body tree as a child
    root = ET.Element("mujoco")
    
   
    # Add the dependency includes as child elements of the root
    for include_path in dependency_includes:
        include_elem = ET.Element("include")
        include_elem.set("file", include_path)
        root.append(include_elem)
    
    
    for body_tree in body_trees:
        root.append(body_tree)

    # Create the .mjcf file
    tree = ET.ElementTree(root)
    tree.write(file_path)

    # Optionally, pretty-print the .mjcf file
    # Note: this requires the 'xml.dom.minidom' module
    # from xml.dom import minidom
    # xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    # with open(file_path, "w") as f:
    #     f.write(xml_str)

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
            
def edit_body_pos(body_list, pos_offset):
    bodies = []
    for body in body_list:
        body.set('name', body.attrib.get('name')+str(pos_offset))
        pos = body.get('pos').split()
        new_pos = [ pos_offset, float(pos[1]), float(pos[2])]
        body.set('pos', ' '.join(map(str, new_pos)))
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

def add_bodies_to_tree(tree, body_list):
    # get the worldbody element
    worldbody = tree.find('worldbody')

    # add each body to the worldbody
    for body in body_list:
        worldbody.append(body)

    return tree
#main_env = main_env_file_name
#poses = [0 , 0.5]
def build_env(main_env_path,poses,sec_files,main_pos):
    #load main env xml file
    #load to random envs files
    #pick the body and independents sections in a dict
    #edit the body x position of both games
    #add the two games, independents to the main file,
    # save the file to the same dir
    #return the file name
    out_file_name = ''
    bodies_names = []
    secondary_envs_names = []
    mjcf_dir = '../sawyer_xyz_multi/mjcfs'
    multi_envs_dir = 'metaworld/envs/assets_v2/sawyer_xyz_multi/'
    mjcfs_save_dir = os.path.join(multi_envs_dir,'mjcfs')
    xml_dir   = os.path.split(main_env_path)[0]
    main_env_name = os.path.split(main_env_path)[1].split('.')[0]
    main_tree = ET.parse(main_env_path)
    

  
    secondary_envs_trees = []
    for file in sec_files:
        secondary_envs_trees.append(ET.parse(os.path.join(xml_dir,file)))
        secondary_env_name = file.split('.')[0][7:]
        secondary_envs_names.append(secondary_env_name)
        bodies_names.append(secondary_env_name+'.mjcf')
    
    envs = defaultdict(list)
    for i,(pos_offset , tree) in enumerate(zip(poses,secondary_envs_trees)):
        bodies_names[i] = str(pos_offset)+bodies_names[i]
        bodies = get_bodies(tree)
        write_mjcf_file(edit_body_pos(bodies,pos_offset), get_tree_includes(tree)  , os.path.join(mjcfs_save_dir,bodies_names[i]))
        envs['includes'] += get_tree_includes(tree)        

    envs['includes'] = kill_duplicates(envs['includes'])
    main_tree = add_includes_to_tree(main_tree,  envs['includes'])
    
    for i , body_name in enumerate(bodies_names):
        main_tree = add_body_to_tree(main_tree,  os.path.join(mjcfs_save_dir,body_name),os.path.join(mjcf_dir,body_name),str(i))

    out_file_name += main_env_name
    for sec_env in secondary_envs_names:
        out_file_name += '-'
        out_file_name += sec_env


    for pos in [main_pos] + poses:
        out_file_name += ','
        out_file_name += str(pos)
        
    out_file_name += '.xml'
    print(out_file_name)

    main_tree.write(os.path.join(multi_envs_dir,out_file_name))
    return out_file_name
    




    

    

