import os
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
import mujoco_py


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
    for include in includes[1:-1]:
        root.append(include)
    for body_tree in body_trees:
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

def add_body_to_tree(main_tree, body_file):
    # Parse the body file
    body_tree = ET.parse(body_file)
    body_root = body_tree.getroot()

    # Find the body element in the body tree
    body_elem = body_root.find('body')

    # Get the name and position of the body
    name = body_elem.attrib['name']
    pos = body_elem.attrib['pos']

    # Create a new body element in the main tree
    new_body_elem = ET.Element('body')
    new_body_elem.attrib['name'] = name
    new_body_elem.attrib['pos'] = pos

    # Add the include element to the new body element
    include_elem = ET.Element('include')
    include_elem.attrib['file'] = body_file
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
        #body.set('name', body.attrib.get('name')+str(pos_offset))
        pos = body.get('pos').split()
        new_pos = [float(pos[0]) + pos_offset, float(pos[1]), float(pos[2])]
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
def build_env(main_env_path,poses,num_envs):
    #load main env xml file
    #load to random envs files
    #pick the body and independents sections in a dict
    #edit the body x position of both games
    #add the two games, independents to the main file,
    # save the file to the same dir
    #return the file name
    out_file_name = 'multi_envs.xml'
    bodies_names = ['first','second']
    xml_dir   = os.path.split(main_env_path)[0]
    print(os.path.split(main_env_path)[0],os.path.split(main_env_path)[1])
    main_tree = ET.parse(main_env_path)

    # List of XML files in the directory
    xml_files = os.listdir(xml_dir)

    secondary_envs_trees = []
    # Choose 2 random files from the 3 picked, and exclude the main env file if picked.
    random_files = random.sample(xml_files, 3)
    for file in random_files:
        if file.endswith(main_env_path): pass
        secondary_envs_trees.append(ET.parse(os.path.join(xml_dir,file)))
        if len(secondary_envs_trees) == num_envs-1: break
    
    envs = defaultdict(list)
    for pos_offset , tree in zip(poses,secondary_envs_trees):
        print(tree)
        bodies = get_bodies(tree)

        write_mjcf_file(edit_body_pos(bodies,pos_offset), get_tree_includes(tree)  , os.path.join(xml_dir,'first.mjcf'))
        #envs['includes'] += get_tree_includes(tree)        
        #envs['bodies']   += edit_body_pos(bodies,pos_offset)

    #envs['includes'] = kill_duplicates(envs['includes'])
    #main_tree = add_includes_to_tree(main_tree,  envs['includes'])
    #main_tree = add_bodies_to_tree(main_tree, envs['bodies'])
    
    print(test_mjcf_file( os.path.join(xml_dir,'first.mjcf')))
    print('__________________________________________')
    main_tree = add_body_to_tree(main_tree,  os.path.join(xml_dir,'first.mjcf'))


    main_tree.write(os.path.join(xml_dir,out_file_name))
    return os.path.join(xml_dir,out_file_name)
    




    

    

