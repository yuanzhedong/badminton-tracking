#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument(
    "-res",
    "--resolution",
    type=str,
    default="720",
    help="Sets the resolution on mono cameras. Options: 800 | 720 | 400",
)
parser.add_argument(
    "-md",
    "--mesh_dir",
    type=str,
    default=None,
    help="Output directory for mesh files. If not specified mesh files won't be saved",
)
parser.add_argument(
    "-lm",
    "--load_mesh",
    default=False,
    action="store_true",
    help="Read camera intrinsics, generate mesh files and load them into the stereo node.",
)
parser.add_argument(
    "-rect",
    "--out_rectified",
    default=False,
    action="store_true",
    help="Generate and display rectified streams",
)
parser.add_argument(
    "-lr",
    "--lrcheck",
    default=False,
    action="store_true",
    help="Better handling for occlusions",
)
parser.add_argument(
    "-e",
    "--extended",
    default=False,
    action="store_true",
    help="Closer-in minimum depth, disparity range is doubled",
)
parser.add_argument(
    "-s",
    "--subpixel",
    default=False,
    action="store_true",
    help="Better accuracy for longer distance, fractional disparity 32-levels",
)
parser.add_argument(
    "-m",
    "--median",
    type=str,
    default="7x7",
    help="Choose the size of median filtering. Options: OFF | 3x3 | 5x5 | 7x7 (default)",
)
parser.add_argument(
    "-d",
    "--depth",
    default=False,
    action="store_true",
    help="Display depth frames",
)
parser.add_argument(
    "-swlr",
    "--swap_left_right",
    default=False,
    action="store_true",
    help="Swap left right frames",
)
parser.add_argument(
    "-a",
    "--alpha",
    type=float,
    default=None,
    help="Alpha scaling parameter to increase FOV",
)
args = parser.parse_args()

RES_MAP = {
    '800': {'w': 1280, 'h': 800, 'res': dai.MonoCameraProperties.SensorResolution.THE_800_P },
    '720': {'w': 1280, 'h': 720, 'res': dai.MonoCameraProperties.SensorResolution.THE_720_P },
    '400': {'w': 640, 'h': 400, 'res': dai.MonoCameraProperties.SensorResolution.THE_400_P }
}
if args.resolution not in RES_MAP:
    exit("Unsupported resolution!")

resolution = RES_MAP[args.resolution]

meshDirectory = args.mesh_dir  # Output dir for mesh files
generateMesh = args.load_mesh  # Load mesh files

outRectified = args.out_rectified  # Output and display rectified streams
lrcheck = args.lrcheck  # Better handling for occlusions
extended = args.extended  # Closer-in minimum depth, disparity range is doubled
subpixel = args.subpixel  # Better accuracy for longer distance, fractional disparity 32-levels
depth = args.depth  # Display depth frames

medianMap = {
    "OFF": dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF,
    "3x3": dai.StereoDepthProperties.MedianFilter.KERNEL_3x3,
    "5x5": dai.StereoDepthProperties.MedianFilter.KERNEL_5x5,
    "7x7": dai.StereoDepthProperties.MedianFilter.KERNEL_7x7,
}
if args.median not in medianMap:
    exit("Unsupported median size!")

median = medianMap[args.median]

print("StereoDepth config options:")
print(f"    Resolution:  {resolution['w']}x{resolution['h']}")
print("    Left-Right check:  ", lrcheck)
print("    Extended disparity:", extended)
print("    Subpixel:          ", subpixel)
print("    Median filtering:  ", median)
print("    Generating mesh files:  ", generateMesh)
print("    Outputting mesh files to:  ", meshDirectory)


def getMesh(calibData):
    M1 = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, resolution[0], resolution[1]))
    d1 = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B))
    R1 = np.array(calibData.getStereoLeftRectificationRotation())
    M2 = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, resolution[0], resolution[1]))
    d2 = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C))
    R2 = np.array(calibData.getStereoRightRectificationRotation())
    mapXL, mapYL = cv2.initUndistortRectifyMap(M1, d1, R1, M2, resolution, cv2.CV_32FC1)
    mapXR, mapYR = cv2.initUndistortRectifyMap(M2, d2, R2, M2, resolution, cv2.CV_32FC1)

    meshCellSize = 16
    meshLeft = []
    meshRight = []

    for y in range(mapXL.shape[0] + 1):
        if y % meshCellSize == 0:
            rowLeft = []
            rowRight = []
            for x in range(mapXL.shape[1] + 1):
                if x % meshCellSize == 0:
                    if y == mapXL.shape[0] and x == mapXL.shape[1]:
                        rowLeft.append(mapYL[y - 1, x - 1])
                        rowLeft.append(mapXL[y - 1, x - 1])
                        rowRight.append(mapYR[y - 1, x - 1])
                        rowRight.append(mapXR[y - 1, x - 1])
                    elif y == mapXL.shape[0]:
                        rowLeft.append(mapYL[y - 1, x])
                        rowLeft.append(mapXL[y - 1, x])
                        rowRight.append(mapYR[y - 1, x])
                        rowRight.append(mapXR[y - 1, x])
                    elif x == mapXL.shape[1]:
                        rowLeft.append(mapYL[y, x - 1])
                        rowLeft.append(mapXL[y, x - 1])
                        rowRight.append(mapYR[y, x - 1])
                        rowRight.append(mapXR[y, x - 1])
                    else:
                        rowLeft.append(mapYL[y, x])
                        rowLeft.append(mapXL[y, x])
                        rowRight.append(mapYR[y, x])
                        rowRight.append(mapXR[y, x])
            if (mapXL.shape[1] % meshCellSize) % 2 != 0:
                rowLeft.append(0)
                rowLeft.append(0)
                rowRight.append(0)
                rowRight.append(0)

            meshLeft.append(rowLeft)
            meshRight.append(rowRight)

    meshLeft = np.array(meshLeft)
    meshRight = np.array(meshRight)

    return meshLeft, meshRight


def saveMeshFiles(meshLeft, meshRight, outputPath):
    print("Saving mesh to:", outputPath)
    meshLeft.tofile(outputPath + "/left_mesh.calib")
    meshRight.tofile(outputPath + "/right_mesh.calib")


def getDisparityFrame(frame, cvColorMap):
    maxDisp = stereo.initialConfig.getMaxDisparity()
    disp = (frame * (255.0 / maxDisp)).astype(np.uint8)
    disp = cv2.applyColorMap(disp, cvColorMap)

    return disp

device = dai.Device()
calibData = device.readCalibration()
print("Creating Stereo Depth pipeline")
pipeline = dai.Pipeline()

camLeft = pipeline.create(dai.node.MonoCamera)
camRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutRight = pipeline.create(dai.node.XLinkOut)
xoutDisparity = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutRectifLeft = pipeline.create(dai.node.XLinkOut)
xoutRectifRight = pipeline.create(dai.node.XLinkOut)

xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

if args.swap_left_right:
    camLeft.setCamera("right")
    camRight.setCamera("left")
else:
    camLeft.setCamera("left")
    camRight.setCamera("right")

for monoCam in (camLeft, camRight):  # Common config
    monoCam.setResolution(resolution['res'])
    # monoCam.setFps(20.0)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(median)  # KERNEL_7x7 default
stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
stereo.setLeftRightCheck(lrcheck)
stereo.setExtendedDisparity(extended)
stereo.setSubpixel(subpixel)

if args.alpha is not None:
    stereo.setAlphaScaling(args.alpha)
    config = stereo.initialConfig.get()
    config.postProcessing.brightnessFilter.minBrightness = 0
    stereo.initialConfig.set(config)

spatialLocationCalculator.inputConfig.setWaitForMessage(False)

for i in range(10):
    config = dai.SpatialLocationCalculatorConfigData()
    config.depthThresholds.lowerThreshold = 200
    config.depthThresholds.upperThreshold = 10000
    config.roi = dai.Rect(dai.Point2f(i*0.1, 0.45), dai.Point2f((i+1)*0.1, 0.55))
    spatialLocationCalculator.initialConfig.addROI(config)


xoutLeft.setStreamName("left")
xoutRight.setStreamName("right")
xoutDisparity.setStreamName("disparity")
xoutDepth.setStreamName("depth")
xoutRectifLeft.setStreamName("rectifiedLeft")
xoutRectifRight.setStreamName("rectifiedRight")

xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

camLeft.out.link(stereo.left)
camRight.out.link(stereo.right)
stereo.syncedLeft.link(xoutLeft.input)
stereo.syncedRight.link(xoutRight.input)
stereo.disparity.link(xoutDisparity.input)
if depth:
    stereo.depth.link(xoutDepth.input)
if outRectified:
    stereo.rectifiedLeft.link(xoutRectifLeft.input)
    stereo.rectifiedRight.link(xoutRectifRight.input)

spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)
spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)


streams = ["left", "right"]
if outRectified:
    streams.extend(["rectifiedLeft", "rectifiedRight"])
streams.append("disparity")
if depth:
    streams.append("depth")

cvColorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
cvColorMap[0] = [0, 0, 0]
print("Creating DepthAI device")

video_writer = cv2.VideoWriter('/home/nvidia/badminton-tracking/logs/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640*3, 480))
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#video_writer = cv2.VideoWriter('/home/nvidia/badminton-tracking/logs/output.avi', fourcc, 20.0, (640*3, 480))
frame_count = 0
with device:
    device.startPipeline(pipeline)

    # Create a receive queue for each stream
    qList = [device.getOutputQueue(stream, 8, blocking=False) for stream in streams]
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    color = (0,200,40)
    fontType = cv2.FONT_HERSHEY_TRIPLEX
    while True:
        final_frames = []
        for q in qList:
            name = q.getName()
            frame = q.get().getCvFrame()
            if name == "depth":
                #frame = frame.astype(np.uint16)
                print(frame)
                depth_downscaled = frame[::4]
                if np.all(depth_downscaled == 0):
                    min_depth = 0  # Set a default minimum depth value when all elements are zero
                else:
                    min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
                max_depth = np.percentile(depth_downscaled, 99)
                depthFrameColor = np.interp(frame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

                spatialData = spatialCalcQueue.tryGet()
                spatialData = [] if spatialData is None else spatialData.getSpatialLocations()
                for depthData in spatialData:
                    roi = depthData.config.roi
                    roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])

                    xmin = int(roi.topLeft().x)
                    ymin = int(roi.topLeft().y)
                    xmax = int(roi.bottomRight().x)
                    ymax = int(roi.bottomRight().y)

                    coords = depthData.spatialCoordinates
                    distance = math.sqrt(coords.x ** 2 + coords.y ** 2 + coords.z ** 2)

                    cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, thickness=2)
                    cv2.putText(depthFrameColor, "{:.1f}m".format(distance/1000), (xmin + 10, ymin + 20), fontType, 0.6, color)
                frame = depthFrameColor

            elif name == "disparity":
                #frame = getDisparityFrame(frame, cvColorMap)
                continue
            if len(frame.shape) < 3:
                frame = np.stack((frame,)*3, axis=-1)
            final_frames.append(frame)
            print(name, frame.shape)

        final_frame = cv2.hconcat(final_frames)
        color = (0,200,40)
        fontType = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(final_frame, f"{frame_count}", (10, 20), fontType, 0.6, color)
        cv2.imshow("final", final_frame)
        video_writer.write(final_frame)
        frame_count += 1
        if cv2.waitKey(1) == ord("q"):
            video_writer.release()
            break
        if frame_count > 1000:
            video_writer.release()
            break
