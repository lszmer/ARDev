from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n-seg.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo11n.onnx'
