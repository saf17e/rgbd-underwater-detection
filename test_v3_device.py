import depthai as dai

pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setFps(30)
q_isp = cam.isp.createOutputQueue(maxSize=4, blocking=False)
q_ctrl = cam.inputControl.createInputQueue()

device = dai.Device()
device.startPipeline(pipeline)
print("Device started with startPipeline!")
