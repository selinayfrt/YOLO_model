from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train( data="C:/Users/90545/Downloads/dataset.yaml", epochs=100, imgsz=800, pretrained=True, save_period=1, plots=True, device=0, workers=0, patience=50)