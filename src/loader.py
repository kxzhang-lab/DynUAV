# This file is used to load the dataset and extract the frames.
import glob
import cv2
import os
import re
import numpy as np
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import pandas as pd

############################ part 1: 加载视频数据，并将视频数据转化为包含每一帧图像的文件夹 ############################
def load_dataset(dataset_path):
    """
      brief: 把包含每一帧的视频文件夹找到或者构建出来
      args: 
        dataset_path: 数据集路径
      return: 视频帧数据和相应的标注文件
    """
    video_paths = glob.glob(f"{dataset_path}/**/*.MP4",recursive=True)
    if not len(video_paths):
        # 查找包含帧图像的文件夹
        frame_folders = search_frame_folders(dataset_path)
    else:
        frame_folders = []
        for video_path in video_paths:
            video_folder = load_video(video_path)
            if not video_folder:
                continue
            frame_folders.append(video_folder)
    return frame_folders
    
def load_video(video_path):
    """
      brief: 加载视频并一帧帧读取
      args: 
        video_path: 视频文件路径
    """
    # 1. 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video file.")
        return None
    
    # 2. 获取有关该视频的详细统计信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 3. 创建文件夹保存视频帧
    video_name = os.path.basename(video_path).split(".")[0]
    video_folder = os.path.join(os.path.dirname(video_path), video_name)
    video_folder = os.path.join(video_folder,"img1")
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
        
    # 4. 如果文件夹中已经有被保存了的视频帧，则不用重复操作了。
    img_files = glob.glob(os.path.join(video_folder,"*.jpg"))
    require_capture = True
    if (len(img_files) == frame_count):
        cap.release()
        require_capture = False
        
    # 4. 如果没有，则读取视频帧并保存
    if require_capture:
        # 设置读取分辨率为原视频分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                print(f"警告：第{frame_num}帧读取时分辨率不符。视频：{width}*{height}，实际：{frame.shape[1]}*{frame.shape[0]}")
            cv2.imwrite(os.path.join(video_folder, f"frame_{frame_num:06d}.jpg"),frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_num += 1
        cap.release()
        print(f"saving {video_name} video into frames is finished!")   
    
    # 5. 考察视频文件夹内有没有保存该视频相关信息的.csv文件
    csv_file = os.path.join(video_folder, "video_info.csv")
    require_save_csv = True
    if os.path.exists(csv_file):
        with open(csv_file, "r") as f:
            lines = f.readlines()
            if len(lines) == 2 \
                and lines[0].strip() == f"frame_count,width,height,fps" \
                and lines[1].strip() == f"{frame_count},{width},{height},{fps}":
                require_save_csv = False
                f.close()
                
    # 5. 在该视频文件夹内创建.csv文件保存视频信息
    if require_save_csv:
        csv_file = os.path.join(video_folder, "video_info.csv")
        with open(csv_file, "w") as f:
            f.write(f"frame_count,width,height,fps\n \
                    {frame_count},{width},{height},{fps}\n")
            f.close()
    return video_folder

### 补充：load_video的CP：把图片合成视频####
def images_to_video(image_folder, video_stem, fps=30):
    """
    将指定文件夹中的图片合成为视频
    :param image_folder: 图片所在文件夹路径
    :param video_stem: 输出视频文件的后缀名
    :param fps: 帧率（frames per second）
    """
    try:
        # 获取并排序图片文件
        images = [img for img in os.listdir(image_folder)
                  if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if not images:
            raise ValueError("未找到任何图片文件")

        images.sort()  # 按文件名排序，确保顺序正确

        # 读取第一张图片，获取尺寸
        first_img_path = os.path.join(image_folder, images[0])
        frame = cv2.imread(first_img_path)
        if frame is None:
            raise ValueError(f"无法读取图片: {first_img_path}")
        height, width, _ = frame.shape

        output_path = image_folder + video_stem
        # 选择编码器（mp4 用 mp4v，avi 用 XVID）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') if video_stem[1:].lower() == 'mp4' \
                 else cv2.VideoWriter_fourcc(*'XVID')

        # 创建 VideoWriter 对象
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for img_name in images:
            img_path = os.path.join(image_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"跳过无法读取的图片: {img_path}")
                continue
            # 如果尺寸不一致，调整大小
            if (img.shape[1], img.shape[0]) != (width, height):
                img = cv2.resize(img, (width, height))
            out.write(img)  # 写入视频帧

        out.release()
        print(f"视频已保存到: {output_path}")

    except Exception as e:
        print(f"发生错误: {e}")


def search_frame_folders(dataset_path):
    """
      brief: 查找包含帧图像的文件夹
      args: 
        dataset_path: 数据集路径
    """
    frame_dirs = set()
    extensions=('jpg', 'jpeg', 'png', 'bmp')
    for root, _, _ in os.walk(dataset_path):
        img_files = []
        for ext in extensions:
            files = glob.glob((os.path.join(root,f"*.{ext}")))
            upper_files = glob.glob((os.path.join(root,f"*.{ext.upper()}")))
            if files:
                img_files.extend(files)
                break
            elif upper_files:
                img_files.extend(upper_files)
                break
        if img_files:
            frame_dirs.add(root)
    return list(frame_dirs)


############################ part 2: 针对UAV123数据集，从标注文件中提取出每个对应的视频 ###############################
# 自定义排序键函数
def get_sort_key(name) -> tuple:
    """生成排序键：(主名称, 后缀类型, 数字部分)"""
    name = os.path.splitext(name)[0]  # 去除扩展名
    
    # 处理分段标注如 car1_1.txt
    if match := re.match(r"^(.+?)_(\d+)$", name):
        base = match.group(1)
        num = int(match.group(2))
        suffix_type = 1  # 分段标注
        return (base, suffix_type, num)
    
    # 处理_s后缀
    if name.endswith('_s'):
        return (name[:-2], 2, 0)  # 类型2表示_s后缀
    
    # 普通情况
    return (name, 0, 0)  # 类型0表示无后缀
        

def extract_video_from_uav123(dataset_path):
    """
      brief: 从标注文件中提取出每个对应的视频
      args:
        dataset_path: 数据集路径
        frame_folders: 包含每个视频帧的文件夹列表
      return: 视频帧数据和相应的标注文件
    """
    # 由于该函数是针对UAV123数据集的，因此可以根据数据集总体路径获得标注文件的路径。
    anno_path = os.path.join(dataset_path, "anno/UAV123")
    anno_files = sorted(glob.glob(os.path.join(anno_path, "*.txt")),
                        key=lambda x: get_sort_key(os.path.basename(x)))
    
    seq_path = os.path.join(dataset_path, "data_seq/UAV123")
    frame_folders = sorted(
        [os.path.join(seq_path, f) for f in os.listdir(seq_path) if os.path.isdir(os.path.join(seq_path, f))],
        key=lambda x: get_sort_key(x)
    )
    
    match_samples = []
    mismatch_samples = []
    # 同步读取每个视频文件夹和标注文件名
    # 要是遇到文件夹名称和标注文件名称相同的，则该视频帧属于该标注文件
    # 要是遇到标注文件名称为**_#.txt并且**就是文件夹名称的，说明该视频被分段标注了。
    # 此时也将该视频分开
    anno_id = 0
    for frame_folder in frame_folders:
        # 注意：判断标注文件列表的下标越界
        if anno_id >= len(anno_files):
            print(f"Warning: No more annotation files for {frame_folder}.")
            break
        
        # 1. 获取视频名称
        video_name = os.path.basename(frame_folder)  # 视频帧文件夹名称
        
        # 2. 匹配标注文件
        anno_file = anno_files[anno_id]
        anno_name = os.path.basename(anno_file).split(".")[0]
        
        # 3. 如果视频名称和标注文件名称相同，则说明该视频帧属于该标注文件
        if video_name == anno_name:
            # 3.1 “视频-标注”组合
            video_anno = {
                "frame_folder":frame_folder,
                "anno_file":anno_file
            }
            # 3.2 考察视频的每一帧是不是和文件里的每个标注对应
            frame_imgs = glob.glob(os.path.join(frame_folder, "*.jpg"))
            anno_number = get_anno_number(anno_file)
            if anno_number == len(frame_imgs):
                # 帧和标注一一对应
                match_samples.append(video_anno)    
            else:
                # 帧和标注不一一对应
                mismatch_samples.append(video_anno)
            # 3.3 切到下一个标注文件
            anno_id += 1
            continue
        
        # 4 处理分段标注的视频
        # 4.1 如果标注文件的名称_后面为数字，则说明该文件夹对应的标注文件是分段的
        is_video_split = anno_name.startswith(video_name + "_") \
            and anno_name.split("_")[-1].isdigit()
        if is_video_split:
            # 4.2 获取该视频中的所有帧图像
            frame_imgs = glob.glob(os.path.join(frame_folder, "*.jpg"))
            # 4.3 把每段标注文件的标注数目累加
            anno_nums = 0  
            annos_of_video = []
            while is_video_split:
                anno_number = get_anno_number(anno_file)
                anno_nums += anno_number
                annos_of_video.append(anno_file)
                
                # 切到下一段标注文件
                anno_id += 1
                anno_file = anno_files[anno_id]
                anno_name = os.path.basename(anno_file).split(".")[0]
                is_video_split = anno_name.startswith(video_name + "_") \
                    and anno_name.split("_")[-1].isdigit()
                    
            # 4.4 构造“视频-标注”组合 
            video_anno = {
                "frame_folder": frame_folder,
                "anno_file":annos_of_video
            }
            # 4.5 考察视频的每一帧是不是和文件里的每个标注对应
            if anno_nums == len(frame_imgs):
                # 帧和标注一一对应
                match_samples.append(video_anno)
            else:
                # 帧和标注不一一对应
                mismatch_samples.append(video_anno)
                
    return match_samples, mismatch_samples        


def get_anno_number(anno_file):
    """
      brief: 
        从标注文件中获取标注的数量
      args:
        anno_file: 标注文件路径
      return: 
        标注数量
    """
    with open(anno_file, "r") as f:
        anno_lines = f.readlines()
        anno_num = len(anno_lines)
        f.close()
    return anno_num


#########################  实验3：样本边界框的统计分析  #######################
def load_UAV123_annos(UAV123_anno,aggre_anno_dir):
    """
      brief:
        加载UAV123数据集的标注文件
      args:
        UAV123_anno: UAV123数据集的标注文件路径
        aggre_anno_dir: 整合以后的新文件标注路径
      return:
        所有的标注文件名称
    """
    # 从标注路径下获取所有的标注文件名称并返回所有标注文件的路径
    UAV_anno_files = glob.glob(os.path.join(UAV123_anno, "*.txt"))
    if not UAV_anno_files:
        raise ValueError(f"No annotation files found in {UAV123_anno}")
    # 对UAV_anno_files文件的后处理
    new_anno_files = []
    anno_id = 0
    while anno_id < len(UAV_anno_files):
        anno_file = UAV_anno_files[anno_id]
        file_name = os.path.basename(anno_file)
        base_name, ext = os.path.splitext(file_name)
        if '_' in base_name and base_name.split('_')[-1].isdigit():
            # 获取主文件名 (如 uav1)
            main_name = base_name.split('_')[0]
            new_file = os.path.join(aggre_anno_dir, f"{main_name}{ext}")   
            all_lines = []
            while anno_id < len(UAV_anno_files) and \
                   os.path.basename(anno_file).startswith(main_name + '_') \
                   and os.path.splitext(os.path.basename(anno_file))[0].split('_')[-1].isdigit():
                with open(anno_file,"r") as f:
                    lines = f.readlines()
                    all_lines.extend(lines)
                    f.close()
                anno_id = anno_id + 1
                anno_file = UAV_anno_files[anno_id]
            with open(new_file,"w") as f:
                f.writelines(all_lines)
                f.close() 
            new_anno_files.append(new_file)
        else:
            new_file = os.path.join(aggre_anno_dir,os.path.basename(anno_file))
            shutil.copy2(anno_file,new_file)
            new_anno_files.append(new_file)
            anno_id = anno_id + 1
    
    return new_anno_files


################################# 实验 4: 针对VIsDrone数据集进行加载 ###############################
def load_annotations_txt(anno_path,column_defs):
    """
    brief:
      安全加载.txt标注文件，自动跳过不完整行。
    args:
      anno_path:标注文件绝对路径
      column_defs:标注定义列
    return:
      返回结构化NumPy数组，各列独立类型
    """
    # 先读取为文本检查完整性
    with open(anno_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # 过滤不完整行（每行应有恰好10个逗号分隔值）
    valid_lines = []
    for line in lines:
        if line.count(',') == len(column_defs) - 1:  # 10列=9个逗号
            # 检查数值有效性（示例：排除含空值的行）
            parts = line.split(',')
            if all(part.strip() for part in parts):
                valid_lines.append(line)
    
    # 转换为结构化数组
    dtype = np.dtype(column_defs)
    try:
        data = np.genfromtxt(
            valid_lines, 
            delimiter=',',
            dtype=dtype,
            filling_values=0  # 缺失值填充为0（根据需求调整）
        )
        
        names = [x[0] for x in column_defs]
        if 'frame' not in names and 'id' not in names:
            # 将原始数据转换为元组列表
            original_data = [tuple(x) for x in data]
            # 添加frame编号
            enhanced_data = [(i,) + (0,)+ row for i, row in enumerate(original_data)]
            # 转换为新结构化数组
            new_dtype = np.dtype([('frame', 'i4')] + [('id', 'i4')] + column_defs)
            data = np.array(enhanced_data, dtype=new_dtype)
            
        # print(f"Loaded {len(data)} valid annotations (skipped {len(lines)-len(valid_lines)} invalid lines)")
        return data
    except Exception as e:
        print(f"Error loading annotations: {e}")
        return np.array([], dtype=dtype)


def group_by_column(annotations,col_name):
    """按指定列分组标注"""
    groups = {}
    for anno in annotations:
        key = anno[col_name]
        groups.setdefault(key, []).append(anno)
    return groups


######################################## # 补充：MDMT数据集的标注文件为.xml文件 ######################################
def analyze_xml_categories(xml_files):
    """
    分析多个XML文件中的所有类别
    
    参数:
        xml_files: XML文件路径列表
    
    返回:
        category_mapping: 类别到数字的映射字典
        reverse_mapping: 数字到类别的反向映射字典
    """
    all_categories = set()
    
    # 如果是单个文件，转换为列表
    if isinstance(xml_files, str):
        xml_files = [xml_files]
    
    # 首先收集所有类别
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            for track_item in root.findall("track"):
                label = track_item.attrib.get('label')
                if label:
                    all_categories.add(label)
                    
        except Exception as e:
            print(f"分析文件 {xml_file} 时出错: {e}")
            continue
    
    # 按字母顺序排序以确保一致性
    sorted_categories = sorted(list(all_categories))
    
    # 创建映射字典
    category_mapping = {category: idx for idx, category in enumerate(sorted_categories)}
    reverse_mapping = {idx: category for idx, category in enumerate(sorted_categories)}
    
    return category_mapping, reverse_mapping


def load_annotations_xml(anno_path,column_defs,category_mapping=None):
    """
    brief:
      针对MDMT数据集的.xml文件进行加载。
      该文件的格式为：
         <annotations count="130">
           <track id="0" label="car">
             <box frame="0" occluded="0" outside="0" xbr="725" xtl="672" ybr="656" ytl="574"/>
             <box frame="1" occluded="0" outside="0" xbr="726" xtl="673" ybr="644" ytl="563"/>
             ...
             <box frame="197" occluded="1" outside="0" xbr="898" xtl="879" ybr="6" ytl="0"/>
           </track>
           <track id="1" label="car">
             <box frame="0" occluded="0" outside="0" xbr="775" xtl="738" ybr="300" ytl="252"/>
             <box frame="1" occluded="0" outside="0" xbr="775" xtl="738" ybr="293" ytl="246"/>
             ...
             <box frame="311" occluded="0" outside="0" xbr="28" xtl="4" ybr="1072" ytl="1066"/>
           </track>
         </annotations>
    args:
      anno_path:标注文件绝对路径
      column_defs:标注定义列
      category_mapping:类别映射字典
    return:
      该视频所有标注条目的结构化数组
    """
    # 如果没有提供映射字典，先分析当前文件的类别
    if category_mapping is None:
        temp_mapping, _ = analyze_xml_categories(anno_path)
        category_mapping = temp_mapping
        
    try:
        # 读取XML文件
        tree = ET.parse(anno_path)  # 从文件读取
        root = tree.getroot()
        
        all_boxes = []
        unknown_categories = set()  # 记录未知类别
        
        for track_item in root.findall("track"):
            track_id = int(track_item.attrib["id"])
            track_label = track_item.attrib.get("label")
            
            # 获取类别ID，如果类别不在映射中，使用-1并记录
            if track_label in category_mapping:
                class_id = category_mapping[track_label]
            else:
                class_id = -1
                unknown_categories.add(track_label)
            
            for box_item in track_item.findall("box"):
                # 步骤1：读每个bb的相关信息
                frame = int(box_item.attrib['frame'])
                occluded = int(box_item.attrib['occluded'])
                outside = int(box_item.attrib['outside'])
                xbr = float(box_item.attrib['xbr'])
                xtl = float(box_item.attrib['xtl'])
                ybr = float(box_item.attrib['ybr'])
                ytl = float(box_item.attrib['ytl'])
                
                # 步骤2：求当前bb的宽度和高度
                width = xbr - xtl
                height = ybr - ytl
                
                # 确保宽度和高度为正数
                if width < 0:
                    width = abs(width)
                    xtl = xbr
                if height < 0:
                    height = abs(height)
                    ytl = ybr
                    
                # 步骤3：按照结构化数组的顺序构造元组
                box_data = (
                    frame,          # frame
                    track_id,       # id
                    xtl,            # x (左上角x)
                    ytl,            # y (左上角y)
                    width,          # w (宽度)
                    height,         # h (高度)
                    outside,        # out-of-view
                    occluded,       # occlusion
                    class_id        # class
                )
                all_boxes.append(box_data)
        
        # 如果有未知类别，打印警告
        if unknown_categories:
            print(f"警告: 发现未知类别: {unknown_categories}")
    
        # 创建结构化数组
        if all_boxes:
            structured_array = np.array(all_boxes, dtype=column_defs)
        else:
            # 如果没有数据，创建空数组
            structured_array = np.array([], dtype=column_defs)
        return structured_array
    
    except Exception as e:
        print(f"分析文件 {anno_path} 时出错: {e}")
        return None


def save_structured_array_to_csv(structured_array, output_path,file_name, reverse_mapping=None):
    """
    将结构化数组保存为CSV文件，可选择包含类别名称
    """
    if len(structured_array) == 0:
        print("没有数据可保存")
        return
    # 将结构化数组转换为DataFrame
    df = pd.DataFrame(structured_array)
    
    # 如果需要添加类别名称列
    if reverse_mapping is not None:
        # 添加类别名称列
        df['class_name'] = df['class'].map(lambda x: reverse_mapping.get(x, f"unknown_{x}"))
    
    # 保存到CSV
    df.to_csv(str(output_path / f"{file_name}.csv"), index=False, float_format='%.2f')
    print(f"数据已保存到 {output_path}，共 {len(df)} 行")
        

###################################### 实验5：获取关于视频路径、分辨率等信息  ##################################
def get_image_files(video_path, extensions=('jpg', 'jpeg', 'png', 'bmp')):
    """
    brief:
        查询某视频文件夹下的所有视频图像
    args:
        video_path: 视频文件夹路径
        extensions: 图像后缀集合
    """
    img_files = []
    for ext in extensions:
        # 构建glob模式，注意os.path.join不能直接处理元组
        pattern = os.path.join(video_path, f"**/*.{ext}")
        files = glob.glob(pattern,recursive=True)
        
        # 如果需要处理大写扩展名
        pattern_upper = os.path.join(video_path, f"**/*.{ext.upper()}")
        upper_files = glob.glob(pattern_upper,recursive=True)
        
        if files:
            img_files.extend(files)
            break
        elif upper_files:
            img_files.extend(upper_files)
            break
    
    return img_files


def get_video_name(video_path,dataset_name):
    """
    brief: 通过视频文件夹路径和数据集名称获取视频名称
    Args:
        video_path (Path): 视频文件夹路径
        dataset_name (str): 数据集名称
    """
    if dataset_name in ('DanceTrack', 'MOT17', 'MOT20', 'DBT70', 'OURS', 'SportsMOT'):
        video_name = f"{Path(video_path).parts[-2]}"
    else:
        video_name = f"{Path(video_path).stem}"
    return video_name


def get_video_resolution(anno_file,dataset_name):
    """
    brief:
      通过标注文件获取相应视频的分辨率
    Args:
      anno_file (str): 标注文件的绝对路径
      dataset_name (str): 数据集的名称
    return:
      res: (dict):分辨率
      video_path: (str):视频文件夹路径
    """
    anno_name = os.path.basename(anno_file).split(".")[0]
    if dataset_name == "VisDrone":
        video_path = Path(anno_file).parents[1] / "sequences" / anno_name
    elif dataset_name == "UAV123":   
        video_path = Path(anno_file).parents[1] / "Dataset_UAV123" / "data_seq" /"UAV123" /anno_name
    elif dataset_name == "UAVDT":
        video_name = re.match(r"^[^_]*", anno_name).group()
        video_path = Path(anno_file).parents[2] / "UAV-benchmark-M" / video_name
    elif dataset_name == "MDMT":
        dir_name = ["test","train","val"]
        is_match=False
        for name in dir_name:
            path = str(anno_file).replace("new_xml",name).split('.')[0]
            if os.path.exists(path):
                video_path = path
                is_match = True
                break
        if not is_match:
            raise ValueError(f"{dataset_name}:{anno_file} don't match a video.")
    elif dataset_name == "MOT20" or dataset_name == "MOT17" \
        or dataset_name == "DanceTrack" or dataset_name == "SportsMOT":
        video_path = Path(anno_file).parents[1] / f"img1"
    elif dataset_name == "DBT70":
        video_path = Path(anno_file).parents[0] / f"img"
    elif dataset_name == "NAT2021" or dataset_name == "UAVTrack112":
        video_path = Path(anno_file).parents[1] / "data_seq" / anno_name
    elif dataset_name == "OURS":
        video_path = Path(anno_file).parents[1] / "img1"
        
    # 判断：标注文件对应的视频是否存在
    if not os.path.exists(video_path):
        raise ValueError(f"{video_path} is not exists.")
    
    img_files = get_image_files(video_path)
    if not img_files:
        raise ValueError(f"there is no images in {video_path}")
    
    img = cv2.imread(img_files[0])
    height, width, _ = img.shape
    res = {"H":height,"W":width}
    return res


def get_frame_dleta(anno,anno_file,dataset_name):
        """
        brief:
          获取视频文件路径和实际图像帧序号与标注帧序号的增量。
        args:
          anno:当前视频文件的标注条目数组。
          anno_file:标注文件路径
          dataset_name:数据集名称
        """
        # 初始化
        video_path,frame_dleta = "",0
        # 1. 获得视频路径
        anno_name = os.path.basename(anno_file).split(".")[0]
        if dataset_name == "VisDrone":
            video_path = Path(anno_file).parents[1] / "sequences" / anno_name
        elif dataset_name == "UAV123":    
            video_path = Path(anno_file).parents[1] / "Dataset_UAV123" / "data_seq" /"UAV123" /anno_name
        elif dataset_name == "UAVDT":
            video_name = re.match(r"^[^_]*", anno_name).group()
            video_path = Path(anno_file).parents[2] / "UAV-benchmark-M" / video_name
        elif dataset_name == "MDMT":
            dir_name = ["test","train","val"]
            is_match=False
            for name in dir_name:
                path = str(anno_file).replace("new_xml",name).split('.')[0]
                if os.path.exists(path):
                    video_path = path
                    is_match = True
                    break
            if not is_match:
                raise ValueError(f"{dataset_name}:{anno_file} don't match a video.")
        elif dataset_name == "MOT20" or dataset_name == "MOT17" \
            or dataset_name == "DanceTrack" or dataset_name == "SportsMOT" :
            video_path = Path(anno_file).parents[1] / "img1"
        elif dataset_name == "DBT70":
            video_path = Path(anno_file).parents[0] / f"img"
        elif dataset_name == "NAT2021" or dataset_name == "UAVTrack112":
            video_path = Path(anno_file).parents[1] / "data_seq" / anno_name
        elif dataset_name == "OURS":
            video_path = Path(anno_file).parents[1] / "img1"
            
        # 判断：标注文件对应的视频是否存在
        if not os.path.exists(video_path):
            raise ValueError(f"{video_path} is not exists.")    
        
        # 2. 考察该标注文件中的帧序号最小是从0开始标还是从1开始标
        if 'frame' not in anno.dtype.names:
            frame_dleta = 0
            video_path = video_path
            frame_dleta = frame_dleta
        
        # 3. 获取最小标注序号  
        frame_col = np.unique(anno["frame"])
        min_frame = np.min(frame_col)
        
        # 4. 获取视频中图像的实际最小序号
        # 4.1 获取视频文件夹下所有帧图像
        img_files = get_image_files(video_path)
        # 4.2 获取图像名称中最小的序号
        min_image_index = find_min_image_number(img_files)
            
        # 4.3 实际图像的序号相对标注帧序号的增量计算
        frame_dleta = min_image_index - min_frame
        
        return video_path,frame_dleta


def extract_number_partition(filename:str):
    """
    brief:
        在图像文件名中提取出图像序号
    param:
        filename为图像文件名
    return:
        文件名中包含的序号
    """
    # 处理扩展名
    name = filename.lower()
    if '.' in name:
        name = name.rsplit('.', 1)[0]
    
    # 纯数字情况
    if name.isdigit():
        return int(name)
    
    # 使用partition分割
    for prefix in ['img', 'frame_']:
        if name.startswith(prefix):
            number_part = name.partition(prefix)[2]
            if number_part.isdigit():
                return int(number_part)
    
    return None


def find_min_image_number(files):
    """
    brief:直接遍历查找视频文件夹中最小图像序号，避免创建列表和排序
    params:files为视频文件夹中的所有图片路径
    return:最小图像序号
    """
    min_number = float('inf')  # 初始化为无穷大
    
    for filename in files:
        number = extract_number_partition(os.path.basename(filename))
        if number is not None and number < min_number:
            min_number = number
    
    # 如果没有找到有效文件
    if min_number == float('inf'):
        return None
    
    return min_number

    
def get_image_name_pattern(dataset_name,frame_dleta,frame_num):
    """
    brief: 根据数据集的名称，带显示的帧序号和标注信息返回图像名称。
    """
    if "VisDrone" == dataset_name: 
        img_pattern = f"{(frame_num + frame_dleta):07d}.jpg"
    elif "UAV123" == dataset_name: 
        img_pattern = f"{(frame_num + frame_dleta):06d}.jpg"
    elif "UAVDT" == dataset_name:
        img_pattern = f"img{(frame_num + frame_dleta):06d}.jpg"
    elif "MDMT" == dataset_name:
        img_pattern = f"{(frame_num + frame_dleta):08d}.jpg"
    elif "MOT20" == dataset_name or "MOT17" == dataset_name or "SportsMOT" == dataset_name:
        img_pattern = f"{(frame_num + frame_dleta):06d}.jpg"
    elif "DanceTrack" == dataset_name:
        img_pattern = f"{(frame_num + frame_dleta):08d}.jpg"
    elif "DBT70" == dataset_name:
        img_pattern = f"{(frame_num + frame_dleta):05d}.jpg"
    elif "NAT2021" == dataset_name:
        img_pattern = f"{(frame_num + frame_dleta):06d}.jpg"
    elif "UAVTrack112" == dataset_name:
        img_pattern = f"{(frame_num + frame_dleta):05d}.jpg"
    elif "OURS" == dataset_name:
        img_pattern = f"frame_{(frame_num + frame_dleta):06d}.png"
    return img_pattern
    

######################################## 实验 6:将结构化数组中关于边界框的列取出来，转化为普通数组 #####################################
def get_bbox(anno,column_defs):
    """
    brief:
      把结构化数组的标注条目中关于边界框的信息都抽取出来，并放到普通数组中。
    args:
        anno ([[np.void],[np.void],..,[np.void]]]): 保存所有标注信息的结构化数组。
        column_defs:结构化数组的列标题。
    return:
        返回只包含边界框的普通数组。
    """
    # 1. 判断标注信息是不是完整地包含了边界框的信息
    first_elements = [col[0] for col in column_defs]
    has_xywh = all(
        key in first_elements and len(anno[key]) > 0
        for key in ['x', 'y', 'w', 'h']
    )
    if not has_xywh:
        print("标注条目不包含完整的边界框信息")
        return None
    # 2. 构造仅包含边界框信息的普通数组
    bbox_anno = np.stack([anno['x'],anno['y'],anno['w'],anno['h']],axis=1)
    return bbox_anno


################################### 实验7:分割数据集 ###################################
def get_dataset_split_paths(dataset_root,anno_files):
    """
    基于关键字分类的标注文件路径拆分
    
    参数:
        dataset_root: 数据集根目录路径
        anno_files: 所有标注文件路径
    返回:
        tuple: (split_dict, has_split)
        - split_dict: 包含三个键值对的字典：
        {
            'train': 训练集路径列表
            'val': 验证集路径列表
            'test': 测试集路径列表
        }
        - split_dirs: 字典，包含各子集的根目录路径 
        {
            'train': '/path/to/train',
            'val': '/path/to/val',
            'test': '/path/to/test'
        }    
        - has_split: 是否检测到数据集划分
    """
    # 1. 定义分类关键字
    CATEGORY_KEYWORDS = {
        'train': ['train', 'training', 'trn'],
        'val': ['val', 'valid', 'validation'],
        'test': ['test', 'testing', 'eval', 'evaluation']
    }
    
    # 2. 初始化结果字典
    split_dict = {category: [] for category in CATEGORY_KEYWORDS.keys()}
    split_dirs = {category: [] for category in CATEGORY_KEYWORDS.keys()}
    
    # 3. 先判断可不可以分段
    has_split = all(
        [
            any(
                # 1. 判断每一条文件是否有“训练”或者“验证”或者“测试”的相关关键字
                [
                    # 2. 只要该文件属于“训练”或者“验证”或者“测试”的任意一个，那么该文件可拆分。
                    any(
                        [
                            # 3. 只要相关子集中的任何一个关键字出现在路径里，那么路径属于该子集。
                            pat in anno_file.lower() for pat in pattern
                        ]
                    ) for _,pattern in CATEGORY_KEYWORDS.items()
                ]
            ) for anno_file in anno_files
        ]
    )
    
    # 4. 如果不可拆，则直接返回空字典核不可拆标志。
    if not has_split:
        return (split_dict,split_dirs,has_split)
    
    # 5. 如果可拆，则做进一步的拆分。
    for anno_file in anno_files:
        lower_file = anno_file.lower()
        for category,pattern in CATEGORY_KEYWORDS.items():
            if any([part in lower_file for part in pattern]):
                split_dict[category].append(anno_file)
                
    # 6.返回每个子集的总体路径
    annotated_dirs = set()
    for category in CATEGORY_KEYWORDS.keys():
        if not split_dict[category]:
            continue
            
        # 取第一个文件路径作为示例来提取根目录
        sample_path = split_dict[category][0]
        path_parts = Path(sample_path).parts
        
        # 找到包含类别关键字的目录部分
        for i, part in enumerate(path_parts):
            lower_part = part.lower()
            if any(keyword in lower_part or lower_part in keyword \
                for keyword in CATEGORY_KEYWORDS[category]):
                # 从开始到匹配部分的路径
                split_dir = str(Path(*path_parts[:i+1]))
                if split_dir not in split_dirs[category]:
                    split_dirs[category].append(split_dir)
                annotated_dirs.add(split_dir)
                break
            
    # 7. 查找无标注的子集路径
    # 遍历数据集根目录下的所有子目录
    for dir_path in Path(dataset_root).rglob('*'):
        if not dir_path.is_dir():
            continue
            
        dir_str = str(dir_path)
        lower_dir = dir_str.lower()
        
        # 检查是否匹配任何类别关键字
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(keyword in lower_dir for keyword in keywords):
                # 检查是否已经存在父路径被记录
                is_parent_recorded = any(
                    dir_str.startswith(recorded_dir + "/") or 
                    dir_str.startswith(recorded_dir + "\\")
                    for recorded_dir in split_dirs[category] + list(annotated_dirs)
                )
                # 如果这个目录还没有被记录，并且它也不存在被记录的父路径，就把它记录上。
                not_record = dir_str not in split_dirs[category] and dir_str not in annotated_dirs
                if not_record and not is_parent_recorded:
                    split_dirs[category].append(dir_str)
                    
    # 8. 对每个类别的路径列表进行排序，使有标注的路径在前面
    for category in split_dirs:
        # 将路径列表分成有标注的和无标注的
        annotated = [d for d in split_dirs[category] if d in annotated_dirs]
        unannotated = [d for d in split_dirs[category] if d not in annotated_dirs]
        split_dirs[category] = annotated + unannotated
            
    return (split_dict,split_dirs,has_split)
    