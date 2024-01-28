import pygltflib
import struct
import json

class MeshVertex:
    def __init__(self, position, normal, tex_coord, weights, joints):
        self.position = position
        self.normal = normal
        self.tex_coord = tex_coord
        self.weights = weights
        self.joints = joints

    def to_dict(self):
        return {
            "Position": self.position,
            "Normal": self.normal,
            "TexCoords": self.tex_coord,
            "Tangent": None,  # No Tangent data in MeshVertex class
            "Bitangent": None,  # No Bitangent data in MeshVertex class
            "BoneIDs": self.joints,
            "Weights": self.weights
        }

class Bone:
    def __init__(self, name: str, id:int, parent_id: int | None, offset_matrix):
        self.name = name
        self.id = id
        if parent_id == None:
            parent_id = -1
        self.parent_id = parent_id
        self.offset_matrix = offset_matrix

    def to_dict(self):
        return {
            "Id": self.id,
            "Name": self.name,
            "ParentName": None,  # No ParentName data in Bone class
            "ParentId": self.parent_id,
            "Offset": self.offset_matrix
        }

class Skeleton:
    def __init__(self, name: str, bones) -> None:
        self.name = name
        self.bones = bones

    def to_dict(self):
        return {
            "Bones": [bone.to_dict() for bone in self.bones]
        }

class Mesh:
    def __init__(self, vertices, indices):
        self.vertices = vertices
        self.indices = indices

    def to_dict(self):
        return {
            "name": "name",  # Replace with the actual mesh name
            "Vertices": [vertex.to_dict() for vertex in self.vertices],
            "Indices": self.indices
        }

class Model:
    def __init__(self, meshes, skeleton: Skeleton | None) -> None:
        self.meshes = meshes
        self.skeleton = skeleton

    def to_dict(self):
        return {
            "Meshes": [mesh.to_dict() for mesh in self.meshes],
            "Skeleton": self.skeleton.to_dict() if self.skeleton else None
        }

def load_primitive_attribute(index: int, count: int, gltf: pygltflib.GLTF2) -> list:
    accessor = gltf.accessors[index]
    bufferView = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[bufferView.buffer]
    data = gltf.get_data_from_buffer_uri(buffer.uri)
    component_type = accessor.componentType
    values = []
    # component types
    # BYTE = 5120
    # UNSIGNED_BYTE = 5121
    # SHORT = 5122
    # UNSIGNED_SHORT = 5123
    # UNSIGNED_INT = 5125
    # FLOAT = 5126
    if component_type == pygltflib.UNSIGNED_INT:
        bpv = 4
        format = "<I"
    elif component_type == pygltflib.UNSIGNED_SHORT:
        bpv = 2
        format = "<H"
    elif component_type == pygltflib.FLOAT:
        bpv = 4
        format = "<f"
    elif component_type == pygltflib.UNSIGNED_BYTE:
        bpv = 1
        format = "<B"
    else:
        raise RuntimeError(f"Unknown component type {component_type}")

    # update with count
    bpv *= count
    for i in range(count - 1):
        format += format[-1]
    for i in range(accessor.count):
        index = bufferView.byteOffset + accessor.byteOffset + i*bpv
        d = data[index:index+bpv]
        v = struct.unpack(format, d)
        if len(v) == 1:
            v = v[0]
        values.append(v)
    return values

def get_inverse_bind_matrix(skin: pygltflib.Skin, gltf: pygltflib.GLTF2, index: int):
    # Retrieve inverse bind matrices
    if skin.inverseBindMatrices is not None:
        accessor = gltf.accessors[skin.inverseBindMatrices]
        bufferView = gltf.bufferViews[accessor.bufferView]
        buffer = gltf.buffers[bufferView.buffer]
        inverse_bind_matrices_data = gltf.get_data_from_buffer_uri(buffer.uri)

        matrix_size = 4 * 4  # 4x4 matrix with each element as a float
        start_index = (index * matrix_size * 4) + bufferView.byteOffset + accessor.byteOffset
        end_index = ((index + 1) * matrix_size * 4) + bufferView.byteOffset + accessor.byteOffset
        matrix_data = inverse_bind_matrices_data[start_index:end_index]
        # convert the data into a 4x4 matrix
        matrix = [[struct.unpack('<f', matrix_data[j:j+4])[0] for j in range(k, k+16, 4)] for k in range(0, matrix_size * 4, 16)]
        return matrix


def process_bone_relationships(bone_relationship : dict, gltf: pygltflib.GLTF2, skin: pygltflib.Skin):
    bones = []
    # assuming bones already ordered
    for i, joint in enumerate(skin.joints):
        name= gltf.nodes[joint].name
        parent_id = None
        parent_index = None
        for key in bone_relationship.keys():
            values = bone_relationship[key]
            for val in values:
                if joint == val:
                    parent_id = key
                    break
            if parent_id is not None:
                break
        if parent_id is not None:
            for index, j in enumerate(skin.joints):
                if j == parent_id:
                    parent_index = index
                    break

        # bones.append(Bone(
        #     name = name,
        #     index = i,
        #     id = joint,
        #     parent_id = parent_id,
        #     parent_index = parent_index,
        #     inverse_bind_matrix=get_inverse_bind_matrix(skin, gltf, i),
        # ))
        bones.append(Bone(
            name = name,
            id = i,
            parent_id = parent_index,
            offset_matrix=get_inverse_bind_matrix(skin, gltf, i),
        ))
    return bones

def process_skin(skin: pygltflib.Skin, gltf: pygltflib.GLTF2) :
    bone_relationship = {}
    for joint in skin.joints:
        node = gltf.nodes[joint]
        bone_relationship[joint] = node.children
    bones = process_bone_relationships(bone_relationship, gltf, skin)
    skeleton = Skeleton(skin.name, bones)
    return skeleton

def extract_meshes(mesh: pygltflib.Mesh, gltf : pygltflib.GLTF2, skeleton: Skeleton | None) :
    meshes = []
    for primitive_index, primitive in enumerate(mesh.primitives):
        vertices = []
        
        positions = load_primitive_attribute(primitive.attributes.POSITION, 3, gltf)
        if primitive.attributes.TEXCOORD_0 != None:
            tex_coords = load_primitive_attribute(primitive.attributes.TEXCOORD_0, 2, gltf)
        if primitive.attributes.NORMAL != None:
            normals = load_primitive_attribute(primitive.attributes.NORMAL, 3, gltf)
        if primitive.attributes.WEIGHTS_0 != None and skeleton != None:
            weights = load_primitive_attribute(primitive.attributes.WEIGHTS_0, 4, gltf)
        if primitive.attributes.JOINTS_0 != None and skeleton != None:
            joints = load_primitive_attribute(primitive.attributes.JOINTS_0, 4, gltf)
        indices = load_primitive_attribute(primitive.indices, 1, gltf)
        
        # create vertices
        for i in range(len(positions)):
            vertices.append(MeshVertex(
                position=positions[i],
                tex_coord=tex_coords[i],
                weights=weights[i],
                normal=normals[i],
                joints=joints[i]
            ))
        meshes.append(Mesh(vertices=vertices, indices=indices))
    return meshes

# load file
def load_model_gltf(path: str):
    gltf = pygltflib.GLTF2().load(filename)

    mesh = gltf.meshes[0]
    skeleton = None
    meshes = []
    for node_index, node in enumerate(gltf.nodes):
        if node.mesh != None:
            # process skin and get skeleton
            # process only if skeleton is None
            if node.skin != None and skeleton == None:
                skin = gltf.skins[node.skin]
                skeleton = process_skin(skin, gltf)
            # get and process mesh
            mesh = gltf.meshes[node.mesh]
            me = extract_meshes(mesh, gltf, skeleton)
            for m in me:
                meshes.append(m)
                

    return Model(meshes, skeleton)

def save_model_to_file(model, filename):
    with open(filename, 'w') as file:
        json.dump(model.to_dict(), file, indent=2)


filename = "res/animated_cube.glb"
save_model_to_file(load_model_gltf(filename), "mesh_data.json")



