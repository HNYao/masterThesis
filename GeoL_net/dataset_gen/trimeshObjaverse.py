import trimesh
file_path = "dataset/obj/mesh/pencil/pencil_0006_orange/mesh.obj"
# 加载模型
mesh = trimesh.load(file_path, force="mesh")

# 检查是否加载了材质和纹理
print(mesh.visual)

# 保存模型
mesh.export(file_path)