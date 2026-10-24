危险化学品车辆检测数据集，用于目标检测
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0fd8c034bb744872a8f79e5962e272ed.png)

用于危化车辆检测/油罐车检测"
4360+图像，jpg图像与txt标签一一对应，4类
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a27fe754388f416fa41a04230d443352.png)

"大卡车、油罐车、大巴车、小汽车"，按照3:1:1比例，

直接点用于yolo![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/99516d4cf443475c97dc927cb948eac9.png)
YOLOv8为例，提供一个简化的指南来帮助你开始训练模型、评估性能以及进行推理。

### 数据准备

首先确保你的数据集已经按照YOLO格式组织好，即每个图像都有一个对应的`.txt`文件，该文件包含边界框信息。每行代表一个对象，格式如下：

```
class_id center_x center_y width height
```

所有值都是相对于图像尺寸归一化后的浮点数。确保你的数据集按照以下结构组织：

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

并且有一个配置文件`data.yaml`来定义数据集路径和类别信息：

```yaml
train: ./dataset/images/train/
val: ./dataset/images/val/
test: ./dataset/images/test/

nc: 4  # 类别数量
names: ['大卡车', '油罐车', '大巴车', '小汽车']  # 类别名
```

请根据实际的数据集路径修改上述`data.yaml`文件中的路径。

### 安装依赖项

确保你已经安装了YOLOv8及其依赖项。可以从Ultralytics的GitHub仓库获取YOLOv8:

```bash
pip install ultralytics
```

### 模型训练

使用以下Python脚本开始训练过程：

```python
from ultralytics import YOLO

def main_train():
    # 加载YOLOv8模型，'yolov8n.yaml'表示使用nano版本，你可以选择其他尺寸如's', 'm', 'l', 'x'
    model = YOLO('yolov8n.yaml')  # 或者直接加载预训练权重，例如'ultralytics/yolov8n.pt'
    
    results = model.train(
        data='./path/to/data.yaml',  # 替换为你的data.yaml路径
        epochs=300,  # 根据需要调整训练周期数
        imgsz=640,  # 图像尺寸
        batch=16,  # 批大小，根据你的硬件条件调整
        project='./runs/detect',
        name='hazardous_vehicles_detection',
        optimizer='SGD',
        device='0',  # 使用GPU编号，'0'表示第一个GPU
        save=True,
        cache=True,
    )

if __name__ == '__main__':
    main_train()
```

### 模型评估

训练完成后，可以使用验证集对模型进行评估：

```python
from ultralytics import YOLO

model = YOLO('./runs/detect/hazardous_vehicles_detection/weights/best.pt')
metrics = model.val(data='./path/to/data.yaml')
print(metrics.box.map)  # 输出mAP值等指标
```

### 推理与可视化

加载训练好的模型进行推理，并可视化结果：

```python
import cv2
from PIL import Image
from ultralytics import YOLO

model = YOLO('./runs/detect/hazardous_vehicles_detection/weights/best.pt')

def detect_vehicles(image_path):
    results = model.predict(source=image_path)
    img = cv2.imread(image_path)
    for result in results:
        boxes = result.boxes.numpy()
        for box in boxes:
            r = box.xyxy
            x1, y1, x2, y2 = int(r[0]), int(r[1]), int(r[2]), int(r[3])
            label = result.names[int(box.cls)]
            confidence = box.conf
            if confidence > 0.5:  # 设置置信度阈值
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绘制矩形框
                cv2.putText(img, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return img

# 示例调用
result_image = detect_vehicles('your_test_image.jpg')
Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)).show()  # 使用PIL显示图像
```

以上步骤涵盖了从数据准备到模型训练、评估和推理的基本流程。请根据实际情况调整代码中的细节，比如路径设置、超参数配置等。
