"""
   Copyright (C) 2017 Autodesk, Inc.
   All rights reserved.

   Use of this software is subject to the terms of the Autodesk license agreement
   provided at the time of installation or download, or which otherwise accompanies
   this software in either electronic or hard copy form.

"""

import sys
import scipy
import os

import scipy.interpolate

from scipy.spatial.transform import Rotation as R
import argparse

from lib.common.utils import load_config

import json
import math
import numpy as np

import smplx
import torch
import trimesh

sys.path.append("./externals/fbx-python-sdk/samples")

import FbxCommon
from FbxCommon import *
from fbx import *


def STAR_SMPLX():
    cache_path = "./data/init_body/data.npz"
    data = np.load(cache_path)
    faces_list = [torch.as_tensor(data["dense_faces"])]
    dense_lbs_weights = torch.as_tensor(data["dense_lbs_weights"])
    unique_list = [data["unique"]]
    smplx_vt = data["vt"]
    smplx_ft = data["ft"]

    example = trimesh.load(
        OBJ_PATH,
        process=False,
        maintain_order=True,
    )
    vertices = example.vertices
    normals = example.vertex_normals

    vertices = np.load(CANO_VERTEX_PATH)  # should in cano T-pose

    faces = example.faces

    keypoint = np.load(CANO_JOINT_PATH)

    smplx_partents = [
        -1,
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        9,
        9,
        12,
        13,
        14,
        16,
        17,
        18,
        19,
        15,
        15,
        15,
        20,
        25,
        26,
        20,
        28,
        29,
        20,
        31,
        32,
        20,
        34,
        35,
        20,
        37,
        38,
        21,
        40,
        41,
        21,
        43,
        44,
        21,
        46,
        47,
        21,
        49,
        50,
        21,
        52,
        53,
    ]

    return (
        vertices,
        faces,
        keypoint,
        dense_lbs_weights,
        smplx_partents,
        smplx_vt,
        smplx_ft,
        normals,
    )


# Map texture over sphere.
# ref: https://forums.autodesk.com/t5/fbx-forum/can-i-modify-the-existing-fbx-by-fbx-sdk-with-importing-a-new/m-p/10473420
def MapTexture(pSdkManager, pNurbs):
    lTexture = FbxFileTexture.Create(pNurbs.GetMesh(), "defaultMat")

    # The texture won't be displayed if node shading mode isn't set to FbxNode.eTEXTURE_SHADING.
    pNurbs.SetShadingMode(FbxNode.EShadingMode.eTextureShading)

    # Set texture properties.
    lTexture.SetFileName(TEXTURE_PATH)  # Resource file is in current directory.
    lTexture.SetTextureUse(FbxTexture.ETextureUse.eStandard)
    lTexture.SetMappingType(FbxTexture.EMappingType.eCylindrical)
    lTexture.SetMaterialUse(FbxFileTexture.EMaterialUse.eModelMaterial)

    lMaterial = pNurbs.GetSrcObject(FbxCriteria.ObjectType(FbxSurfacePhong.ClassId), 0)

    if lMaterial:
        lMaterial.Diffuse.ConnectSrcObject(lTexture)


# Map material over sphere.
# ref: https://github.com/FXTD-ODYSSEY/MayaScript/blob/master/FBXSDK/samples/ExportScene02/ExportScene02.py#L31
def MapMaterial(pSdkManager, pNurbs):
    lMaterial = FbxSurfacePhong.Create(pSdkManager, "defaultMat")

    lBlack = FbxDouble3(0.0, 0.0, 0.0)

    lMaterial.Emissive.Set(lBlack)
    lMaterial.Ambient.Set(lBlack)
    lMaterial.Specular.Set(lBlack)
    lMaterial.TransparencyFactor.Set(0.0)
    lMaterial.Shininess.Set(0.0)
    lMaterial.ReflectionFactor.Set(0.0)

    # Create LayerElementMaterial on Layer 0
    lLayerContainer = pNurbs.GetMesh()
    lLayerElementMaterial = lLayerContainer.GetLayer(0).GetMaterials()

    if not lLayerElementMaterial:
        lLayerElementMaterial = FbxLayerElementMaterial.Create(lLayerContainer, "")
        lLayerContainer.GetLayer(0).SetMaterials(lLayerElementMaterial)

    # The material is mapped to the whole Nurbs
    lLayerElementMaterial.SetMappingMode(FbxLayerElement.EMappingMode.eAllSame)

    # And the material is avalible in the Direct array
    lLayerElementMaterial.SetReferenceMode(FbxLayerElement.EReferenceMode.eDirect)
    pNurbs.AddMaterial(lMaterial)


def CreateScene(pSdkManager, pScene):
    # Create scene info
    lSceneInfo = FbxDocumentInfo.Create(pSdkManager, "SceneInfo")
    lSceneInfo.mTitle = "SMPL-X"
    lSceneInfo.mSubject = "SMPL-X model with weighted skin"
    lSceneInfo.mAuthor = "ExportScene01.exe sample program."
    lSceneInfo.mRevision = "rev. 1.0"
    lSceneInfo.mKeywords = "weighted skin"
    lSceneInfo.mComment = "no particular comments required."
    pScene.SetSceneInfo(lSceneInfo)

    smplxMaleMesh = CreateMesh(pSdkManager, "Mesh")

    lMeshNode = FbxNode.Create(pScene, "meshNode")

    lControlPoints = smplxMaleMesh.GetControlPoints()

    lMeshNode.SetNodeAttribute(smplxMaleMesh)

    lMeshNode.SetShadingMode(FbxNode.EShadingMode.eTextureShading)

    # lMeshNode.SetShadingMode(FbxNode.eTextureShading)

    MapMaterial(pSdkManager, lMeshNode)
    MapTexture(pSdkManager, lMeshNode)

    lSkeletonRoot = CreateSkeleton(pSdkManager, "Skeleton")

    pScene.GetRootNode().AddChild(lMeshNode)
    pScene.GetRootNode().AddChild(lSkeletonRoot)

    lSkin = FbxSkin.Create(pSdkManager, "")
    LinkMeshToSkeleton(lSdkManager, lMeshNode, lSkin)
    AddShape(pScene, lMeshNode)
    AnimateSkeleton(pSdkManager, pScene, lSkeletonRoot)


def AddShape(pScene, node):
    lBlendShape = FbxBlendShape.Create(pScene, "BlendShapes")

    # shapeInfo = open("vertexLoc.txt", "r")
    for j in range(0, 1):
        lBlendShapeChannel = FbxBlendShapeChannel.Create(
            pScene, "ShapeChannel" + str(j)
        )
        lShape = FbxShape.Create(pScene, "Shape" + str(j))
        lShape.InitControlPoints(VertexNum)
        for i in range(0, VertexNum):
            lShape.SetControlPointAt(FbxVector4(0, 0, 0), i)
        lBlendShapeChannel.AddTargetShape(lShape)
        lBlendShape.AddBlendShapeChannel(lBlendShapeChannel)
    node.GetMesh().AddDeformer(lBlendShape)


# Create a cylinder centered on the Z axis.
def CreateMesh(pSdkManager, pName):
    # preparation
    lMesh = FbxMesh.Create(pSdkManager, pName)
    lMesh.InitControlPoints(VertexNum)
    lControlPoints = lMesh.GetControlPoints()

    for i in range(0, VertexNum):
        locX = smplx_vertices[i][0]
        locY = smplx_vertices[i][1]
        locZ = smplx_vertices[i][2]

        vertexLoc = FbxVector4(float(locX), float(locY), float(locZ))
        lControlPoints[i] = vertexLoc

    j = 0

    for i in range(0, FaceNum):

        fragIndex1 = smplx_faces[i][0]
        fragIndex2 = smplx_faces[i][1]
        fragIndex3 = smplx_faces[i][2]

        lMesh.BeginPolygon(i)  # Material index.
        lMesh.AddPolygon(fragIndex1)
        lMesh.AddPolygon(fragIndex2)
        lMesh.AddPolygon(fragIndex3)
        # Control point index.
        lMesh.EndPolygon()

    for i in range(0, VertexNum):
        lMesh.SetControlPointAt(lControlPoints[i], i)

    # Set the normals on Layer 0.
    lLayer = lMesh.GetLayer(0)
    if lLayer == None:
        lMesh.CreateLayer()
        lLayer = lMesh.GetLayer(0)

    # specify normals per control point.
    # For compatibility, we follow the rules stated in the
    # layer class documentation: normals are defined on layer 0 and
    # are assigned by control point.
    normLayer = FbxLayerElementNormal.Create(lMesh, "")

    normLayer.SetMappingMode(FbxLayerElement.EMappingMode.eByControlPoint)
    normLayer.SetReferenceMode(FbxLayerElement.EReferenceMode.eDirect)
    for i in range(0, VertexNum):
        locX = smplx_normals[i][0]
        locY = smplx_normals[i][1]
        locZ = smplx_normals[i][2]
        normLoc = FbxVector4(float(locX), float(locY), float(locZ))
        normLayer.GetDirectArray().Add(normLoc)
    lLayer.SetNormals(normLayer)

    ####
    # Create UV for Diffuse channel
    lUVDiffuseLayer = FbxLayerElementUV.Create(lMesh, "DiffuseUV")
    lUVDiffuseLayer.SetMappingMode(FbxLayerElement.EMappingMode.eByPolygonVertex)
    lUVDiffuseLayer.SetReferenceMode(FbxLayerElement.EReferenceMode.eIndexToDirect)

    lLayer.SetUVs(lUVDiffuseLayer, FbxLayerElement.EType.eTextureDiffuse)

    for i in range(0, smplx_vt.shape[0]):
        uu, vv = smplx_vt[i][0], smplx_vt[i][1]
        # set vt
        lVectors = FbxVector2(uu, 1 - vv)
        lUVDiffuseLayer.GetDirectArray().Add(lVectors)

    # Now we have set the UVs as eINDEX_TO_DIRECT reference and in eBY_POLYGON_VERTEX  mapping mode
    # we must update the size of the index array.
    lUVDiffuseLayer.GetIndexArray().SetCount(FaceNum * 3)

    # Create polygons. Assign texture and texture UV indices.
    # for each face, we have
    for i in range(0, FaceNum):
        # we won't use the default way of assigning textures, as we have
        # textures on more than just the default (diffuse) channel.
        lMesh.BeginPolygon(-1, -1, False)

        fvv = smplx_ft[i]

        # Now we have to update the index array of the UVs for diffuse, ambient and emissive
        for j in range(3):
            lUVDiffuseLayer.GetIndexArray().SetAt(i * 3 + j, fvv[j])

        lMesh.EndPolygon()

    return lMesh


# create 55 skeletons for SMPL-X model
def CreateSkeleton(pSdkManager, pName):

    lSkeletonRootAttribute = FbxSkeleton.Create(lSdkManager, "Root")
    lSkeletonRootAttribute.SetSkeletonType(FbxSkeleton.EType.eLimbNode)
    lSkeletonRootAttribute.Size.Set(JointsSize)
    lSkeletonRoot = FbxNode.Create(lSdkManager, "Root")
    lSkeletonRoot.SetNodeAttribute(lSkeletonRootAttribute)

    lSkeletonRoot.LclTranslation.Set(
        FbxDouble3(
            float(smplx_joints[0][0]),
            float(smplx_joints[0][1]),
            float(smplx_joints[0][2]),
        )
    )

    nodeDict[0] = lSkeletonRoot
    locDict = {
        0: (
            float(smplx_joints[0][0]),
            float(smplx_joints[0][1]),
            float(smplx_joints[0][2]),
        )
    }

    for i in range(1, SkelNum):

        skeletonName = Num2Joints[i]
        skeletonAtrribute = FbxSkeleton.Create(lSdkManager, skeletonName)
        skeletonAtrribute.SetSkeletonType(FbxSkeleton.EType.eLimbNode)
        skeletonAtrribute.Size.Set(JointsSize)
        skeletonNode = FbxNode.Create(lSdkManager, skeletonName)
        skeletonNode.SetNodeAttribute(skeletonAtrribute)
        nodeDict[i] = skeletonNode

        locDict[i] = (
            float(smplx_joints[i][0]),
            float(smplx_joints[i][1]),
            float(smplx_joints[i][2]),
        )

        skeletonFather = int(smplx_partents[i])

        fatherNode = nodeDict[skeletonFather]
        skeletonNode.LclTranslation.Set(
            FbxDouble3(
                float(float(smplx_joints[i][0]) - float(locDict[skeletonFather][0])),
                float(float(smplx_joints[i][1]) - float(locDict[skeletonFather][1])),
                float(float(smplx_joints[i][2]) - float(locDict[skeletonFather][2])),
            )
        )

        fatherNode.AddChild(skeletonNode)

    return lSkeletonRoot


def LinkMeshToSkeleton(pSdkManager, pMeshNode, lSkin):

    for i in range(0, SkelNum):
        skeletonNode = nodeDict[i]
        skeletonName = skeletonNode.GetName()
        skeletonNum = Joints2Num[str(skeletonName)]

        skeletonWeightsInfo = smplx_lbs_weight[:, skeletonNum]

        skeletonCluster = FbxCluster.Create(pSdkManager, "")
        skeletonCluster.SetLink(skeletonNode)
        skeletonCluster.SetLinkMode(FbxCluster.ELinkMode.eNormalize)

        for j in range(0, VertexNum):
            skeletonCluster.AddControlPointIndex(j, float(skeletonWeightsInfo[j]))

        # Now we have the Mesh and the skeleton correctly positioned,
        # set the Transform and TransformLink matrix accordingly.
        lXMatrix = FbxAMatrix()
        lScene = pMeshNode.GetScene()
        if lScene:
            lXMatrix = lScene.GetAnimationEvaluator().GetNodeGlobalTransform(pMeshNode)
        skeletonCluster.SetTransformMatrix(lXMatrix)
        lScene = skeletonNode.GetScene()
        if lScene:
            lXMatrix = lScene.GetAnimationEvaluator().GetNodeGlobalTransform(
                skeletonNode
            )
        skeletonCluster.SetTransformLinkMatrix(lXMatrix)

        # Add the clusters to the Mesh by creating a skin and adding those clusters to that skin.
        # After add that skin.
        lSkin.AddCluster(skeletonCluster)

    pMeshNode.GetNodeAttribute().AddDeformer(lSkin)


def AnimateSkeleton(pSdkManager, pScene, pSkeletonRoot):

    lTime = FbxTime()
    lKeyIndex = 0
    lRoot = pSkeletonRoot
    # First animation stack.
    lAnimStackName = "json generated Animation"
    lAnimStack = FbxAnimStack.Create(pScene, lAnimStackName)

    # The animation nodes can only exist on AnimLayers therefore it is mandatory to
    # add at least one AnimLayer to the AnimStack. And for the purpose of this example,
    # one layer is all we need.
    lAnimLayer = FbxAnimLayer.Create(pScene, "Base Layer")
    lAnimStack.AddMember(lAnimLayer)

    fps = 20

    STAR_pose = np.load(POSE_PATH)

    lTime.SetGlobalTimeMode(FbxTime.EMode.eCustom, fps)
    lTime.SetGlobalTimeProtocol(FbxTime.EProtocol.eFrameCount)

    # Read Rot info from Json
    for i in range(len(STAR_pose)):
        # body pose
        for j in range(22):
            skeletonNode = nodeDict[j]

            lCurveX = skeletonNode.LclRotation.GetCurve(lAnimLayer, "X", True)
            lCurveY = skeletonNode.LclRotation.GetCurve(lAnimLayer, "Y", True)
            lCurveZ = skeletonNode.LclRotation.GetCurve(lAnimLayer, "Z", True)
            lTime.SetFramePrecise(i)

            if lCurveX:
                lCurveX.KeyModifyBegin()
                lKeyIndex = lCurveX.KeyAdd(lTime)[0]
                lCurveX.KeySetValue(lKeyIndex, math.degrees(STAR_pose[i][j][0]))

                lCurveX.KeySetInterpolation(
                    lKeyIndex, FbxAnimCurveDef.EInterpolationType.eInterpolationCubic
                )

                lCurveX.KeyModifyEnd()
            if lCurveY:
                lCurveY.KeyModifyBegin()
                lKeyIndex = lCurveY.KeyAdd(lTime)[0]
                lCurveY.KeySetValue(lKeyIndex, math.degrees(STAR_pose[i][j][1]))

                lCurveY.KeySetInterpolation(
                    lKeyIndex, FbxAnimCurveDef.EInterpolationType.eInterpolationCubic
                )

                lCurveY.KeyModifyEnd()
            if lCurveZ:
                lCurveZ.KeyModifyBegin()
                lKeyIndex = lCurveZ.KeyAdd(lTime)[0]
                lCurveZ.KeySetValue(lKeyIndex, math.degrees(STAR_pose[i][j][2]))

                lCurveZ.KeySetInterpolation(
                    lKeyIndex, FbxAnimCurveDef.EInterpolationType.eInterpolationCubic
                )

                lCurveZ.KeyModifyEnd()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    # parser.add_argument('--mesh', type=str, required=True, help="mesh template, must be obj format")
    parser.add_argument("--text", default=None, help="text prompt")
    parser.add_argument("--negative", default="", help="negative text prompt")
    parser.add_argument("--description", default="", help="exp save folder")
    parser.add_argument("--t2m_model", default="mdiffuse", type=str, help="t2m models")

    args = parser.parse_args()

    cfg = load_config(args.config, "configs/default.yaml")

    cfg.merge_from_list(
        [
            "text",
            args.text,
            "negative",
            args.negative,
        ]
    )

    cfg.freeze()

    text_split = cfg.text.split(", he/she")
    if len(text_split) == 1:
        OBJ_TEXT = text_split[0]
        MOTION_TEXT = "A person is dancing."
    else:
        OBJ_TEXT, MOTION_TEXT = text_split[0], text_split[1]
        MOTION_TEXT = "A person" + MOTION_TEXT

    EXP_DESCRIPTION = args.description

    EXP_SAVE_PATH = os.path.join(
        cfg.training.workspace, cfg.name, OBJ_TEXT, EXP_DESCRIPTION
    )

    OBJ_PATH = "%s/mesh/mesh.obj" % EXP_SAVE_PATH

    CANO_VERTEX_PATH = "%s/mesh/v_cano_dense.npy" % EXP_SAVE_PATH

    CANO_JOINT_PATH = "%s/mesh/keypoint_ori.npy" % EXP_SAVE_PATH

    TEXTURE_PATH = "%s/mesh/mesh_albedo.png" % EXP_SAVE_PATH

    POSE_PATH = "%s/results/motion/%s_%s.npy" % (EXP_SAVE_PATH, OBJ_TEXT, MOTION_TEXT)

    SAMPLE_FILENAME = "%s/results/fbx/%s_%s.fbx" % (
        EXP_SAVE_PATH,
        OBJ_TEXT,
        MOTION_TEXT,
    )

    os.makedirs(SAMPLE_FILENAME, exist_ok=True)

    JointsSize = 1.0
    SkeletonWeights = []
    nodeDict = {}
    Num2Joints = {
        5: "R_Calf",
        8: "R_Foot",
        22: "Jaw",
        23: "L_Eye",
        6: "Spine1",
        18: "L_ForeArm",
        9: "Spine2",
        1: "L_Thigh",
        52: "R_Thumb1",
        54: "R_Thumb3",
        53: "R_Thumb2",
        12: "Neck",
        21: "R_Hand",
        15: "Head",
        24: "R_Eye",
        35: "L_Ring2",
        34: "L_Ring1",
        20: "L_Hand",
        16: "L_UpperArm",
        39: "L_Thumb3",
        38: "L_Thumb2",
        37: "L_Thumb1",
        50: "R_Ring2",
        51: "R_Ring3",
        49: "R_Ring1",
        27: "L_Index3",
        26: "L_Index2",
        25: "L_Index1",
        46: "R_Pinky1",
        17: "R_UpperArm",
        31: "L_Pinky1",
        3: "Spine",
        14: "R_Shoulder",
        42: "R_Index3",
        41: "R_Index2",
        36: "L_Ring3",
        40: "R_Index1",
        19: "R_ForeArm",
        10: "L_Toes",
        45: "R_Middle3",
        44: "R_Middle2",
        43: "R_Middle1",
        7: "L_Foot",
        32: "L_Pinky2",
        33: "L_Pinky3",
        28: "L_Middle1",
        30: "L_Middle3",
        29: "L_Middle2",
        47: "R_Pinky2",
        0: "Root",
        48: "R_Pinky3",
        2: "R_Thigh",
        13: "L_Shoulder",
        4: "L_Calf",
        11: "R_Toes",
    }

    Joints2Num = {
        "R_Calf": 5,
        "R_Foot": 8,
        "Jaw": 22,
        "L_Eye": 23,
        "Spine1": 6,
        "L_ForeArm": 18,
        "Spine2": 9,
        "L_Thigh": 1,
        "R_Thumb1": 52,
        "R_Thumb3": 54,
        "R_Thumb2": 53,
        "Neck": 12,
        "R_Hand": 21,
        "Head": 15,
        "R_Eye": 24,
        "L_Ring2": 35,
        "L_Ring1": 34,
        "L_Hand": 20,
        "L_UpperArm": 16,
        "L_Thumb3": 39,
        "L_Thumb2": 38,
        "L_Thumb1": 37,
        "R_Ring2": 50,
        "R_Ring3": 51,
        "R_Ring1": 49,
        "L_Index3": 27,
        "L_Index2": 26,
        "L_Index1": 25,
        "R_Pinky1": 46,
        "R_UpperArm": 17,
        "L_Pinky1": 31,
        "Spine": 3,
        "R_Shoulder": 14,
        "R_Index3": 42,
        "R_Index2": 41,
        "L_Ring3": 36,
        "R_Index1": 40,
        "R_ForeArm": 19,
        "L_Toes": 10,
        "R_Middle3": 45,
        "R_Middle2": 44,
        "R_Middle1": 43,
        "L_Foot": 7,
        "L_Pinky2": 32,
        "L_Pinky3": 33,
        "L_Middle1": 28,
        "L_Middle3": 30,
        "L_Middle2": 29,
        "R_Pinky2": 47,
        "Root": 0,
        "R_Pinky3": 48,
        "R_Thigh": 2,
        "L_Shoulder": 13,
        "L_Calf": 4,
        "R_Toes": 11,
    }

    (
        smplx_vertices,
        smplx_faces,
        smplx_joints,
        smplx_lbs_weight,
        smplx_partents,
        smplx_vt,
        smplx_ft,
        smplx_normals,
    ) = STAR_SMPLX()

    VertexNum = smplx_vertices.shape[0]
    FaceNum = smplx_faces.shape[0]
    SkelNum = 55

    # Prepare the FBX SDK.
    (lSdkManager, lScene) = FbxCommon.InitializeSdkObjects()

    # Create the scene.
    lResult = CreateScene(lSdkManager, lScene)

    if lResult == False:
        print("\n\nAn error occurred while creating the scene...\n")
        lSdkManager.Destroy()
        sys.exit(1)

    lSdkManager.GetIOSettings().SetBoolProp(EXP_FBX_EMBEDDED, True)
    lFileFormat = lSdkManager.GetIOPluginRegistry().GetNativeWriterFormat()

    # Save the scene.
    # The example can take an output file name as an argument.
    lResult = FbxCommon.SaveScene(
        lSdkManager, lScene, SAMPLE_FILENAME, pFileFormat=0, pEmbedMedia=True
    )

    if lResult == False:
        print("\n\nAn error occurred while saving the scene...\n")
        lSdkManager.Destroy()
        sys.exit(1)

    # Destroy all objects created by the FBX SDK.
    lSdkManager.Destroy()

    print(f"fbx saved at {SAMPLE_FILENAME}")

    sys.exit(0)
