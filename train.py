from ultralytics import YOLO

def main():
    model = YOLO('yolo11n-cls.pt')
    model.train(
        data='./data',
        epochs=50,
        imgsz=224,
        batch=32,
        name='monkey',
        device=0
    )

if __name__ == '__main__':
    main()
