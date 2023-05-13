import xml.etree.ElementTree as ET


def get_xml_includes2(filename):
    root = ET.parse(filename).getroot()
    includes = []
    for child in root:
        if child.tag == "include":
            includes.append(ET.tostring(child, encoding="unicode").strip())
            #includes.append(child.get('file'))
        else:
            break # stop after first non-include element
    return includes


def get_tree_includes(filename):
    root = ET.parse(filename).getroot()
    #root = tree.getroot()
    includes = root.findall('include')
    return includes
def get_bodies(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    worldbody = root.find('worldbody')
    bodies = worldbody.findall('body')
        
    return bodies


def edit_body_pos(body_list, x_offset=0, y_offset=0):
   
    bodies = []
    for body in body_list:
        pos = body.get('pos').split()
        new_pos = [float(pos[0]) + x_offset, float(pos[1]) + y_offset, float(pos[2])]
        body.set('pos', ' '.join(map(str, new_pos)))
        bodies.append(body)

    return bodies


def add_includes_to_xml_file(xml_file_path, includes):
   

    # Load the xml file into an element tree object
    tree = ET.parse(xml_file_path)
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

    # Write the modified xml file back to disk
    tree.write(xml_file_path)


#bodies = get_bodies('metaworld/envs/assets_v2/sawyer_xyz/sawyer_soccer.xml')
#print(edit_body_pos(bodies,1,1))
ret = get_tree_includes('metaworld/envs/assets_v2/sawyer_xyz/sawyer_soccer.xml')
print(ret)
for r in ret:
    print(r.attrib)