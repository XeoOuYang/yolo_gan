import os

from PIL import Image

cwd = os.path.dirname(os.path.abspath(__file__))
from ultralytics import YOLO

def run_best_model(model_path: str, data_set: list[str], **kwargs):
    """
    :param model_path: *.pt权重文件
    :param source:
    :return:
    """
    if kwargs is None: kwargs = {}
    kwargs['show'] = True
    kwargs['save'] = True
    kwargs['exist_ok'] = True
    return YOLO(model_path).predict(source=data_set, **kwargs)

def run_model_with_sahi(model_path, source, **kwargs):
    from sahi.predict import get_sliced_prediction
    from sahi import AutoDetectionModel
    if kwargs is None: kwargs = {}
    detection_model = AutoDetectionModel.from_pretrained(model_type='yolov8', model_path=model_path, **kwargs)
    return get_sliced_prediction(
        image=source, detection_model=detection_model,
        slice_width=256, slice_height=256,
        overlap_width_ratio=0.1, overlap_height_ratio=0.1
    )

if __name__ == '__main__':
    # 最后一次结果
    # val_weight = os.path.join(cwd, 'out', 'detect', 'best.pt')
    # 指定train结果
    val_weight = os.path.join(cwd, 'runs', 'detect', 'train', 'weights', 'best.pt')
    test_path = os.path.join(cwd, 'resource', 'B0104.jpg')
    # test_path = mosaic_magic(test_path, ratio=0.05)
    result = run_best_model(val_weight, [test_path])[0]
    image_path = os.path.join(cwd, result.save_dir, 'B0104.jpg')
    Image.open(image_path).show()
    params = {
        'weight': 'runs/detect/train/weights/best.pt',  # 现在只需要指定权重即可,不需要指定cfg
        'device': 'cuda:0',
        'method': 'HiResCAM',  # GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
        'layer': [10, 12, 14, 16, 18],
        'backward_type': 'class',  # class, box, all
        'conf_threshold': 0.2,  # 0.2
        'ratio': 0.02,  # 0.02-0.1
        'show_box': False,
        'renormalize': False  # 区域突显
    }
    from yolov8heatmap import YoloV8Heatmap
    heatmap = YoloV8Heatmap(**params)
    heatmap('resource/B0104.jpg', 'results')
    # result = run_model_with_sahi(val_weight, test_path)
    # export_dir = os.path.join(cwd, 'out')
    # result.export_visuals(export_dir)
    # image_path = os.path.join(export_dir, "prediction_visual.png")
    # Image.open(image_path).show()