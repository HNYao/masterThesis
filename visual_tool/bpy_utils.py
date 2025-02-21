import bpy


def setMat_pointCloudColoredEmission(mesh, meshColor, ptSize):
    # Create a new material
    mat = bpy.data.materials.new("MeshMaterial")
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree

    # Read vertex attribute (for point cloud color)
    attribute_node = tree.nodes.new("ShaderNodeAttribute")
    attribute_node.attribute_name = "Col"

    # Set brightness/contrast
    BCNode = tree.nodes.new("ShaderNodeBrightContrast")
    BCNode.inputs["Bright"].default_value = meshColor.B
    BCNode.inputs["Contrast"].default_value = meshColor.C
    BCNode.location.x -= 400
    tree.links.new(attribute_node.outputs["Color"], BCNode.inputs["Color"])

    # Replace the Principled BSDF shader with an Emission shader
    emission_node = tree.nodes.new("ShaderNodeEmission")
    emission_node.inputs["Strength"].default_value = (
        1.0  # Adjust emission strength as needed
    )
    tree.links.new(BCNode.outputs["Color"], emission_node.inputs["Color"])

    # Create and connect Material Output node
    output_node = tree.nodes["Material Output"]
    tree.links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])

    # Turn the mesh into a point cloud using a Geometry Nodes modifier
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh

    bpy.ops.object.modifier_add(type="NODES")
    bpy.ops.node.new_geometry_nodes_modifier()
    geo_tree = mesh.modifiers[-1].node_group

    IN = geo_tree.nodes["Group Input"]
    OUT = geo_tree.nodes["Group Output"]

    # Mesh to points conversion
    MESH2POINT = geo_tree.nodes.new("GeometryNodeMeshToPoints")
    MESH2POINT.location.x -= 100
    MESH2POINT.inputs["Radius"].default_value = ptSize

    # Assign the material to the point cloud
    MATERIAL = geo_tree.nodes.new("GeometryNodeSetMaterial")
    geo_tree.links.new(IN.outputs["Geometry"], MESH2POINT.inputs["Mesh"])
    geo_tree.links.new(MESH2POINT.outputs["Points"], MATERIAL.inputs["Geometry"])
    geo_tree.links.new(MATERIAL.outputs["Geometry"], OUT.inputs["Geometry"])

    # Set the material on the points
    MATERIAL.inputs[2].default_value = mat