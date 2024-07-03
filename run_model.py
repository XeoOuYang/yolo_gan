import os.path

from PIL import Image

cwd = os.path.dirname(os.path.abspath(__file__))
from ultralytics import YOLO

from tools import check_cuda_env

def run_model(model_path: str, source: str, **kwargs):
    """
    :param model_path: *.pt权重文件，（*.yaml是模型结构，参考：ultralytics/cfg/models/v8/yolov8.yaml）
    :param source:
    :return:
    """
    if kwargs is None: kwargs = {}
    # kwargs['show'] = True
    kwargs['save'] = True
    kwargs['exist_ok'] = True
    model = YOLO(model_path)
    return model.predict(source=source, **kwargs)


def run_model_with_sahi(model_path, source, **kwargs):
    from sahi.predict import get_sliced_prediction
    from sahi import AutoDetectionModel
    if kwargs is None: kwargs = {}
    detection_model = AutoDetectionModel.from_pretrained(model_type='yolov8', model_path=model_path, **kwargs)
    return get_sliced_prediction(
        image=source, detection_model=detection_model,
        slice_width=384, slice_height=384,
        overlap_width_ratio=0.15, overlap_height_ratio=0.15
    )


if __name__ == '__main__':
    # 测试cuda是否可用
    check_cuda_env()
    # source = os.path.join(cwd, "resource/bus.jpg")
    # source = os.path.join(cwd, "resource/zidane.jpg")
    source = os.path.join(cwd, "resource/B0104.jpg")
    # 检测
    # model_path = os.path.join(cwd, "models", "yolov8x.pt")
    model_path = os.path.join(cwd, "out", "detect", "best.pt")
    # show：表示是否显示结果
    # save：表示是否保存结果
    # exist_ok：表示是否覆盖同名结果
    result = run_model(model_path, source)[0]
    image_path = os.path.join(cwd, result.save_dir, 'B0104.jpg')
    Image.open(image_path).show()
    # result = run_model_with_sahi(model_path, source)
    # export_dir = os.path.join(cwd, 'out')
    # result.export_visuals(export_dir)
    # image_path = os.path.join(export_dir, "prediction_visual.png")
    # Image.open(image_path).show()
    # # 分割
    # model_path = os.path.join(cwd, "models/yolov8x-seg.pt")
    # results = run_model(model_path, source)
    # # 分类
    # model_path = os.path.join(cwd, "models/yolov8x-cls.pt")
    # results = run_model(model_path, source)
    # # 姿态
    # model_path = os.path.join(cwd, "models/yolov8x-pose.pt")
    # results = run_model(model_path, source)
    # # 检测：带旋转方向，比上面检测多一个角度
    # model_path = os.path.join(cwd, "models/yolov8x-obb.pt")
    # results = run_model(model_path, source)
