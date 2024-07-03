import os.path

from PIL import Image

cwd = os.path.dirname(os.path.abspath(__file__))
from ultralytics import YOLO

from tools import chg_suffix, check_cuda_env

def shrink_labels(line_list):
    """
    :param line_list: 分割标签格式 cls,x1,y1,x2,y2,......
    :return: 检测标签格式cls,x,y,w,h
    """
    x_list = line_list[1::2]
    x_list = [float(x) for x in x_list]
    y_list = line_list[2::2]
    y_list = [float(y) for y in y_list]
    max_x = max(x_list)
    min_x = min(x_list)
    c_x = (max_x + min_x) / 2
    max_y = max(y_list)
    min_y = min(y_list)
    c_y = (max_y + min_y) / 2
    w = max_x - min_x
    h = max_y - min_y
    return [line_list[0], str(c_x), str(c_y), str(w), str(h)]

def adjust_class(label_file: str, cls_type: str):
    """
    我们只有1个分类，所以需要调整标签为0
    :param label_file: labeling文件
    :param cls_type: 调整分类标签
    :return:
    """
    adjust_line = []
    # 读取并修改
    with open(label_file, 'r', encoding='utf8', errors='ignore') as fd:
        for line in fd.readlines():
            line_list = line.split()
            # 找到最大的(x, y)，最小的(x, y)，计算中心点和宽高
            if len(line_list) > 5: line_list = shrink_labels(line_list)
            # 矫正错误分类标签
            if line_list[0] != cls_type: line_list[0] = cls_type
            # 复写分类标签
            new_line = ' '.join(line_list)
            adjust_line.append(new_line)
    # 复写文件
    if len(adjust_line) > 0:
        with open(label_file, 'w', encoding='utf8', errors='ignore') as fd:
            fd.write('\n'.join(adjust_line))
    # 修改完成

def check_dataset(base_dir: str):
    """
    检查目录下images、labels是否配对
    :return:
    """
    import glob
    image_dir = os.path.join(base_dir, 'images')
    label_dir = os.path.join(base_dir, 'labels')
    bad_data = []
    for file_name in glob.glob(f'{image_dir}{os.path.sep}*.jpg'):
        label_name = chg_suffix(file_name, suffix='.txt', base_dir=label_dir)
        if not os.path.exists(label_name):
            bad_data.append(file_name)
        else:
            adjust_class(label_name, '0')
    # 输出不正确数据
    if len(bad_data) > 0:
        print(f'check {base_dir} failed. please check {bad_data}')
    else:
        print(f'check {base_dir} successfully')
    # 返回结果
    return len(bad_data) <= 0

def train_mode(model_path: str, data_set: str, epochs: int, **kwargs):
    """
    :param model_path: *.pt权重文件，（*.yaml是模型结构，参考：ultralytics/cfg/models/v8/yolov8.yaml）
    :param data_set: coco8.yaml格式标注数据，包括图片（*.jpg），label文件（*.txt），两者名字相同
    :param epochs:
    :return:
    """
    if kwargs is None: kwargs = {}
    kwargs['save'] = True
    kwargs['exist_ok'] = True
    _model_ = YOLO(model_path)
    return _model_, _model_.train(data=data_set, epochs=epochs, **kwargs)

def run_best_model(model_path: str, data_set: list[str], **kwargs):
    """
    :param model_path: *.pt权重文件
    :param source:
    :return:
    """
    if kwargs is None: kwargs = {}
    # kwargs['show'] = True
    kwargs['save'] = True
    kwargs['exist_ok'] = True
    return YOLO(model_path).predict(source=data_set, **kwargs)

if __name__ == '__main__':
    # 测试cuda是否可用
    check_cuda_env()
    # 检查数据文件
    data_dir = os.path.join(cwd, 'datasets', 'durian')
    train_dir = [
        os.path.join(data_dir, 'train'),
        os.path.join(data_dir, 'valid'),
        os.path.join(data_dir, 'test'),
    ]
    check_result = [check_dataset(base_dir) for base_dir in train_dir]
    data_set = os.path.join(data_dir, 'data.yaml')
    # 通过检查
    if all(check_result):
        # model_path = os.path.join(cwd, "models", "yolov8n.yaml")
        # model_path = os.path.join(cwd, "models", "yolov8.C3RFEM.yaml")
        # model_path = os.path.join(cwd, "models", "yolov8.SEA.yaml")
        # model_path = os.path.join(cwd, "models", "yolov8.CBAM.yaml")
        # model_path = os.path.join(cwd, "models", "yolov8.p4.yaml")
        model_path = os.path.join(cwd, "models", "yolov8m.gold.yaml")
        # model_path = os.path.join(cwd, "models", "yolov8.shuffle.yaml")
        model, result = train_mode(model_path, data_set, 100)
        # 测试验证model
        val_results = model.val()
        # 处理结果
        result_dir = result.save_dir
        if os.path.exists(result_dir):
            train_weights = [
                os.path.join(cwd, result_dir, 'weights', 'best.pt'),
                os.path.join(cwd, result_dir, 'weights', 'last.pt')
            ]
            # # 拷贝到指定目录
            # out_dir = os.path.join(cwd, 'out', 'detect')
            # if not os.path.exists(out_dir): os.makedirs(out_dir)
            # save_weights = [os.path.join(out_dir, os.path.basename(pt_weight)) for pt_weight in train_weights]
            # for src, dst in zip(train_weights, save_weights):
            #     if os.path.exists(dst): os.remove(dst)
            #     shutil.copy(src, dst)
            # 验证结果
            val_weight = train_weights[0]
            test_path = os.path.join(cwd, 'resource', 'B0104.jpg')
            result = run_best_model(val_weight, [test_path])[0]
            image_path = os.path.join(cwd, result.save_dir, 'B0104.jpg')
            Image.open(image_path).show()
    else:
        print(f'please check all data if they are alright')
