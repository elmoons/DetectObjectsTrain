from ultralytics import YOLO

def train():
    model = YOLO("runs/detect/train/weights/last.pt")

    # Train the model using the 'coco8.yaml' dataset for 3 epochs
    results = model.train(data="data.yaml", epochs=300)
    return results

def test():
    model = YOLO('runs/detect/train/weights/last.pt')

    data_yaml = 'data.yaml'

    # Тестирование модели на тестовом наборе данных
    results = model.val(data=data_yaml, split='test', save=True, save_txt=True, save_conf=True)

    # for result in results:
    #     boxes = result.boxes  # Boxes object for bounding box outputs
    #     masks = result.masks  # Masks object for segmentation masks outputs
    #     keypoints = result.keypoints  # Keypoints object for pose outputs
    #     probs = result.probs  # Probs object for classification outputs
    #     obb = result.obb  # Oriented boxes object for OBB outputs
    #     result.show()  # display to screen


if __name__ == '__main__':
    train()
