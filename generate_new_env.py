import os
import random
from xml.etree import ElementTree

# Directory containing the XML files
xml_dir = "metaworld/envs/assets_v2/sawyer_xyz"

# List of XML files in the directory
xml_files = os.listdir(xml_dir)

# Choose 3 random files
random_files = random.sample(xml_files, 3)

# Create a new ElementTree object to hold the merged environment
merged_env = ElementTree.Element("mujoco")

# Add the dependencies to the merged environment
dependencies = [
    "../scene/basic_scene.xml",
    "../objects/assets/window_dependencies.xml",
    "../objects/assets/xyz_base_dependencies.xml"
]
for dep in dependencies:
    ElementTree.SubElement(merged_env, "include", {"file": dep})

worldbody = ElementTree.SubElement(merged_env, "worldbody")

# Loop over the chosen files and append their <worldbody> elements to the merged environment
for file in random_files:
    # Parse the XML file into an ElementTree object
    tree = ElementTree.parse(os.path.join(xml_dir, file))
    
    # Get the root element (should be <mujoco>)
    root = tree.getroot()

    # Get the <worldbody> element
    wb = root.find("worldbody")

    # Append the <worldbody> element to the merged environment
    worldbody.extend(wb.getchildren())

# Create a new ElementTree object to hold the final XML
final_xml = ElementTree.ElementTree(merged_env)

# Write the final XML to a file
final_xml.write(os.path.join(xml_dir,"merged_environment.xml"))