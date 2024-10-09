from ultralytics import YOLO


model = YOLO('runs/detect/train/weights/last.pt')

data_yaml = 'data.yaml'

# Тестирование модели на тестовом наборе данных
results = model.val(data=data_yaml, split='test', save=True, save_txt=True, save_conf=True)



