import depthai as dai

pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.Camera)
# In v3, we can create queue on the node's output:
try:
    q_isp = cam.isp.createOutputQueue()
    q_ctrl = cam.inputControl.createInputQueue()
    print("Queues created directly!")
except Exception as e:
    print(f"Error creating directly: {e}")

with dai.Device(pipeline) as device:
    print("Device started!")
