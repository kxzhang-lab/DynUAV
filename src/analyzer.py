# This file is used to analyze the MOV dataset and generate the final report.
import os
import glob
import cv2
import pandas as pd
import re
import numpy as np
from collections import Counter
from pathlib import Path
import random

from .visualizer import \
  remove_nan_entries, \
  remove_nan_entries_cons, \
  visualize_class_freq,\
  visualize_bbox_dense_video,\
  visualize_bbox_dense_distribution, \
  visualize_bbox_wh, \
  visualize_bbox_wh_change, \
  visualize_bbox_trend, \
  visualize_traj_statistic, \
  visualize_traj, \
  draw_frame_hist, \
  sample_UAV123, \
  visualize_dataset, \
  visualize_bbox
  
from .loader import \
  load_annotations_txt, \
  get_bbox, \
  get_video_resolution, \
  get_frame_dleta, \
  load_dataset, \
  extract_video_from_uav123, \
  get_dataset_split_paths, \
  analyze_xml_categories, \
  load_annotations_xml, \
  get_image_name_pattern, \
  get_image_files, \
  get_video_name
    
############################ part 1: 统计每个视频的帧信息 ############################
# 实验1：计算数据集的帧数统计分布
# 注意：在如下的帧统计特性的开发实验中，数据集的总体路径底下只能包含属于数据集本身的图片。
# 如果需要可视化一些内容并打印到图片，建议保存路径放在数据集总体路径以外。例如：和数据集总体路径的叶子文件夹同级的路径里。
# 不要让这些图片干扰到总体数据集帧图片的统计。
def frame_analysis(dataset_input,dataset_name,save_dir):
    """
    brief:
      计算数据集关于视频总长度的统计特性
    args:
      dataset_input:(list) train/val/test子集或者数据集的整体入口路径列表
      dataset_name:(str) 数据集名称
      save_dir:(Path) 结果保存路径
    """
    # 1. 根据数据集总路径获得每一段视频相应的帧图像文件夹
    frame_folders = []
    for path in dataset_input:
        frame_folders.extend(load_dataset(path))
    
    if not frame_folders:
        return
      
    # 2. 统计每个视频的帧信息
    video_info =collect_frame_information(frame_folders,dataset_name)
    
    save_dir = save_dir / f"video_statistics"
    save_dir.mkdir(parents=True,exist_ok=True)
    
    # 3. 计算关于帧信息的统计信息，并输出到文件中
    all_frame_counts = calc_frame_statistics(video_info, save_dir)
    # 4. 绘制帧数的直方图
    draw_frame_hist(all_frame_counts,save_dir)
    
    
def collect_frame_information(frame_folders,dataset_name):
    """
      brief: 
        This function is used to collect the frame information of each video.
      args:
        frame_folders: a list of frame folders for a video.
        dataset_name: 数据集名称
      return:
        a dictionary containing the frame statistics of the video.
    """
    video_info = {
        "video_name":[],
        "frame_count":[],
        "width":[],
        "height":[],
        "fps":[]
    }
    
    for frame_folder in frame_folders:
        video_name = get_video_name(frame_folder,dataset_name)
        video_info["video_name"].append(video_name)  # 视频名称
        
        csv_finish = False
        csv_file = os.path.join(frame_folder, "video_info.csv")
        if os.path.exists(csv_file):
            with open(csv_file, "r") as f:
                lines = f.readlines()
                if len(lines) == 2:
                    frame_count, width, height, fps = map(int, lines[1].strip().split(","))
                    video_info["frame_count"].append(frame_count)
                    video_info["width"].append(width)
                    video_info["height"].append(height)
                    video_info["fps"].append(fps)
                    csv_finish = True

        if not csv_finish:
            img_files = get_image_files(frame_folder)
            if not img_files:
                print("No image files found in folder: {}".format(frame_folder))
                continue
              
            frame_count = len(img_files)  # 帧数量
            video_info["frame_count"].append(frame_count)
            
            img = cv2.imread(img_files[0])
            height, width, _ = img.shape
            video_info["width"].append(width)  # 宽度
            video_info["height"].append(height)  # 高度
    
    video_info = {k: v for k, v in video_info.items() if len(v) > 0}
    return video_info
  

def calc_frame_statistics(video_info,save_dir):
    save_path = os.path.join(save_dir, "video_statistics.csv")
    pd.DataFrame(video_info).to_csv(save_path, index=False)
    
    all_frame_counts = np.array(video_info["frame_count"]).flatten()
    max_frame_count = np.max(all_frame_counts)
    min_frame_count = np.min(all_frame_counts)
    avg_frame_count = round(np.mean(all_frame_counts))
    sum_frame_count = np.sum(all_frame_counts)
    video_count = len(all_frame_counts)
    
    with open(save_path, "a") as f:
        f.write("\n")
        f.write("len,max,min,avg,sum\n")
        f.write("{},{},{},{},{}\n".format(
          video_count,
          max_frame_count, 
          min_frame_count, 
          avg_frame_count,
          sum_frame_count
          )
        )  
        f.close()
    print("Frame statistics saved to {}".format(save_path))  
    return all_frame_counts  # 返回每个视频的帧数量，用于绘制直方图。
   
   
############################ part 2: 统计每个视频的标注信息 ############################
def bbox_overall_analyze(anno_files,dataset_name,save_dir,column_defs,show_progress):
    """
    brief:
        获取边界框总体分布的统计特性
    args:
        anno_files (list[str]): 每个视频的标注文件集合
        dataset_name (str): 数据集的名称
        save_dir (str): 计算结果的保存路径
        column_defs (list): 构造结构数组的列名称
        show_progress: 展示边界框相对变化可视化的标志位
    """
    
    # 1. 将每个视频文件的标注加载到一个结构化数组中。
    bbox_idx = {}
    if dataset_name == "MDMT":
        category_mapping, _ = analyze_xml_categories(anno_files)
        
    for idx,anno_file in enumerate(anno_files):
        try:
            if dataset_name == "MDMT":
                anno = load_annotations_xml(anno_file,column_defs,category_mapping)
            else:
                anno = load_annotations_txt(anno_file,column_defs)
            # 获取相应视频的分辨率
            res = get_video_resolution(anno_file,dataset_name)
            video_path, frame_dleta = get_frame_dleta(anno,anno_file,dataset_name)
            video_info = {"annos":anno,"res_es":res,"video_path":video_path,"frame_dleta":frame_dleta}
            bbox_idx[idx] = video_info
        except Exception as _:
            # 当前的标注文件未加载到有效信息或者未获取到有效分辨率
            continue
    
    if not bbox_idx:
        print("there is no loaded annotations")
        return 
            
    # 2. 将该结构化数组关于边界框的列都抽出来，并放到一个普通数组中。
    bbox_annos = []
    bbox_res = []  # 按照ID将标注文件拆分后，每个ID对应的分辨率
    bbox_id_idx = {}  # 每个视频的每个id对应的标注和视频路径
    for idx,video_info in bbox_idx.items():
        anno = video_info["annos"]
        bbox_anno = get_bbox(anno,column_defs)
        if bbox_anno is None:
            continue
        
        # 获取anno的id列
        if 'id' in anno.dtype.names and len(np.unique(anno['id'])) > 1:
            # 多目标：按 ID 拆分，每个 ID 对应相同的分辨率
            for obj_id in np.unique(anno['id']):
                mask = anno['id'] == obj_id
                obj_anno = anno[mask]
                bbox_anno = get_bbox(obj_anno,column_defs)
                
                if show_progress:
                    id_info = {
                                "annos":obj_anno,
                                "res_es":video_info["res_es"],
                                "video_path":video_info["video_path"],
                                "frame_dleta":video_info["frame_dleta"]
                              }
                    if dataset_name in ('DanceTrack', 'MOT17', 'MOT20', 'DBT70'):
                        key = f"{Path(id_info["video_path"]).parts[-2]}_{obj_id}"
                    else:
                        key = f"{Path(id_info["video_path"]).stem}_{obj_id}"
                    bbox_id_idx[key] = id_info
                
                bbox_annos.append(bbox_anno)
                bbox_res.append(video_info["res_es"])  # 每个拆分后的目标都关联原视频分辨率     
        # 单目标
        else:
            if show_progress:
                id_info = {
                  "annos":anno,
                  "res_es":video_info["res_es"],
                  "video_path":video_info["video_path"],
                  "frame_dleta":video_info["frame_dleta"]
                }
                if dataset_name in ('DanceTrack', 'MOT17', 'MOT20', 'DBT70'):
                    key = f"{Path(id_info["video_path"]).parts[-2]}"
                else:
                    key = f"{Path(id_info["video_path"]).stem}"
                bbox_id_idx[key] = id_info
            
            bbox_annos.append(bbox_anno)   
            bbox_res.append(video_info["res_es"])
        
    save_dir = save_dir / "anno_statistics"
    save_dir.mkdir(parents=True,exist_ok=True)
            
    # 3. 获得所有ID的边界框相对第一帧的比值
    area_ratios,aspect_changes = get_bbox_changes(bbox_annos)
    visualize_bbox_wh_change(area_ratios,aspect_changes,save_dir)
    if show_progress:
        visualize_bbox_trend(bbox_id_idx,column_defs,dataset_name,save_dir)
    
    # 4. 获取所有边界框的宽高比的总体分布
    # 4.1 将所有的普通数组拼接成一个大数组，该大数组就是数据集的所有边界框信息。
    all_bbox_annos = np.concatenate(bbox_annos)
    # # 4.2 移除所有包含异常值的标注条目
    all_bbox_annos = remove_nan_entries(all_bbox_annos)

    # # 4.3 针对该数据集的所有边界框，进行面积和宽高比的相关统计实验。
    _, aspect_ratios = calc_anno_area_aspectratio(all_bbox_annos)
    # # 5. 获取所有边界框的面积占总面积的比例
    rel_areas = calc_anno_rel_area(bbox_annos,bbox_res)   
    visualize_bbox_wh(rel_areas, aspect_ratios, save_dir)
    

def calc_anno_rel_area(annos,res_es):
    """
    brief:
      获取每个边界框相对总图像的面积比
    Args:
        annos (list(numpy)): 每组标注数组构成的list
        其中：每个元素为shape是(N,4)的数组。每一列依次为x,y,w,h。
        res_es (list): 每组视频相应的分辨率
    """
    assert len(annos) == len(res_es), "视频数目和标注数组数目不符"
    rel_areas = []
    for idx,anno in enumerate(annos):
        anno = remove_nan_entries(anno)
        bbox_wh = anno[:,2:].astype(np.float32)
        area = np.prod(bbox_wh,axis=1)
        
        res = res_es[idx]["H"] * res_es[idx]["W"]
        rel_area = area / res
        rel_areas.append(rel_area)
    rel_areas = np.concatenate(rel_areas)
    return rel_areas
  
    
def calc_anno_area_aspectratio(annos):
    """
      brief: 
        This function is used to calculate the annotation statistics in the whole dataset.
      args:
        annos: numpy of shape (N,4). annos[i]: x,y,w,h
        save_dir: the directory to save the statistics.
      return:
        None
    """
    # 1. 边界框的面积分布直方图
    bbox_wh = annos[:,2:].astype(np.float32)
    area = np.prod(bbox_wh,axis=1)
    
    # 2. 边界框的长宽比分布直方图
    denominators = bbox_wh[:, 1]
    denominators[denominators == 0] = 1e-6  # 避免除以零
    aspect_ratios = bbox_wh[:,0] / denominators
    
    return area, aspect_ratios
    
    
########################### part 3: 统计每个物体下每一帧边界框相对于第一帧的变化分布 ##########################
def get_bbox_changes(bbox_annos):
    """
      brief:
        This function is used to calculate the bbox changes of each object.
      args:
        bbox_annos: a list of annotation boundingbox labels for each video.
      return:
        area_changes: a list of area change rate of each object.
        aspect_ratio_changes: a list of aspect ratio change rate of each object.
    """
    # 1 获取每个标注文件下的所有标注信息。
    area_ratios,aspect_changes = [],[]
    for bbox_anno in bbox_annos:
        area_ratio,aspect_change = get_bbox_change(bbox_anno)
        area_ratios.append(area_ratio)
        aspect_changes.append(aspect_change)
    area_ratios = np.concatenate(area_ratios)
    aspect_changes = np.concatenate(aspect_changes)
    return area_ratios,aspect_changes
  

def get_bbox_change(bbox_anno):
    """
      brief:
        获取每一个ID下边界框相对于第一帧的变化关系
      args:
        bbox_anno: 该视频下所有标注的边界框信息构成的数组。shape:[N,4]。i->[x,y,w,h]
        注：该标注文件保存的是同一个ID的对象在每一帧下的有效边界框信息。
      return:
        area_ratio:每一帧相对于第一帧的边界框面积比。shape:(N-1).N为ID出现的帧数。
        aspect_change:每一帧相对第一帧的边界框长宽比。shape:(N-1).N为ID出现的帧数。
    """
    # 1. 在这些标注信息里，去掉全为NAN的无效标注条目
    bbox_anno = remove_nan_entries(bbox_anno)
    # 2. 针对这些标注信息，计算后面的每一帧相对第一帧的变化
    area,aspect_ratio = calc_anno_area_aspectratio(bbox_anno)
    # 3. 计算相对面积变化和长宽比变化
    area_ratio = area[1:]/area[0]
    aspect_change = aspect_ratio[1:]/aspect_ratio[0]
    
    return area_ratio,aspect_change
  
  
############################## part 4: 针对边界框分布的密集程度做统计 ################################
def bbox_dense_analyze(anno_files,save_dir,column_defs,dataset_name):
    """
    brief: 
      针对数据集，就单帧画面出现的样本数目和ID数目做统计
    args:
      anno_files:所有的数据集文件对应的标注文件
      save_dir:统计结果保存目录
      column_defs:VisDrone数据集的标签定义列
      dataset_name:数据集名称
    """
    # 1. 加载每个视频文件对应的标注信息
    annos = []
    if dataset_name == "MDMT":
        category_mapping, _ = analyze_xml_categories(anno_files)
    for anno_file in anno_files:
        if dataset_name == "MDMT":
            anno = load_annotations_xml(anno_file,column_defs,category_mapping)
        else:
            anno = load_annotations_txt(anno_file,column_defs)
            
        try:
            get_video_resolution(anno_file,dataset_name)
        except Exception as _:
            continue
          
        annos.append(anno)
        
    if not annos:
        print("there is no loaded annotations")
        return
    
    save_dir = save_dir / "anno_statistics"
    save_dir.mkdir(parents=True,exist_ok=True)
    
    # 2. 针对每组视频文件，统计样本数目和ID数目，同时画成条形图并保存。
    bbox_dense_video(annos,save_dir)
    
    # 3. 对单帧画面的样本数、ID数的分布做统计分析
    bbox_dense_distribution(annos,save_dir)
    

def bbox_dense_video(annos,save_dir):    
    """
      brief: 针对每组视频文件，统计样本数目和ID数目，同时画成条形图并保存。
      args: 
        annos:每组视频文件对应的标注数组列表
        save_dir:结果保存路径
    """
    sample_counts,ID_counts = [],[]
    for anno in annos:
        sample_count = len(anno)
        sample_counts.append(sample_count)
        unique_id = np.unique(anno["id"])
        ID_count = len(unique_id)
        ID_counts.append(ID_count)
    visualize_bbox_dense_video(sample_counts,ID_counts,save_dir)
    

def bbox_dense_distribution(annos,save_dir):
    """
      brief:对单帧画面的样本数、ID数做统计分析
      args:
        annos:每组视频文件对应的标注数组列表
        save_dir:结果保存路径 
    """
    # 1. 处理所有视频帧并进行汇总统计
    all_sample_counts = calc_dense_statistic(annos,'frame')
    
    # 2. 绘制比例分布直方图
    visualize_bbox_dense_distribution(all_sample_counts,save_dir)
    
    # 3. 绘制不同class出现频率的条形图
    all_cls_freq = calc_class_distribution(annos)
    visualize_class_freq(all_cls_freq,save_dir)
    
    
def calc_dense_statistic(annos,col_name):
    """
    brief:
      处理所有视频标注文件，汇总统计结果
    args:
      annos:每组视频的标注信息
      col_name:考察标注的哪个维度。例如，"frame"、"id"...
    """
    all_sample_counts = []
    
    # 遍历标注目录下的所有文件
    for anno in annos:
        if len(anno) == 0:
            continue
            
        # 计算当前视频的统计
        sample_counts = calc_bbox_statistic(anno,col_name)
        all_sample_counts.extend(sample_counts)
    
    if col_name == 'id':
        pass
    else:
        return np.array(all_sample_counts)
  
  
def calc_bbox_statistic(annotations,col_name):
    """
    brief:
      计算单帧的样本数和ID数统计
    args:
      annotations:单视频的标注信息
      col_name:考察标注的哪个维度。例如，"frame"、"id"...
    return:
      col_sample_counts: 待分类对象中，每类样本数的数组
    """
    # 获取所有唯一的帧号
    cols = annotations[col_name]
    unique_cols = np.unique(cols)
    
    col_sample_counts = []
    
    for col in unique_cols:
        # 获取当前帧的所有标注
        col_annos = annotations[cols == col]
        
        # 统计样本数(即当前帧的标注行数)
        sample_count = len(col_annos)
        col_sample_counts.append(sample_count)
    
    return np.array(col_sample_counts)
  
  
def calc_class_distribution(annos):
    """
      brief:
        获取视频中的每个类别出现的帧占所有标记的帧的比例
      args:
        annos为每个视频相应的标注信息
      return:
        all_cls_freq:一个字典，存储了每个类别出现的频率。
    """
    all_cls_freq = {}
    for anno in annos:
        all_cls = anno["class"]  # 该文件的标注条目里的类别列
        unique_cls = np.unique(all_cls)  # 该视频中独立的类别
        
        cls_freq = {}
        for cls in unique_cls:
            # 获取当前类别的标注条目
            anno_cls = anno[all_cls == cls]    
            cls_num = len(anno_cls)
            
            cls_freq[cls] = cls_num
            
        all_cls_freq = Counter(all_cls_freq) + Counter(cls_freq)
    return all_cls_freq


############################### part 5: 针对所有id的轨迹间断分布，轨迹持续时间和运动总位移进行统计分析 ####################################
class TrajectoryStatistic:
    def __init__(self,dataset_name,save_dir,is_show):
        """
        brief:
          轨迹统计分析类的初始化
        Args:
            anno_files (list(str)): 所有标注文件的路径构成的列表
            dataset_name (str): 数据集名称
            save_dir (str): 结果保存文件夹路径
            column_defs (numpy): 结构化数组的描述列
        """
        self.anno_ids = []  # 一个字典，里面的元素为(标注文件绝对路径,id)元组
        self.dataset_name = dataset_name  # 保存数据集的名称
        self.save_dir = save_dir  # 保存结果的存储路径
        self.is_show = is_show  # 是否显示每段连续轨迹的若干帧
        self.category_mapping = None  # 类别-序号映射表
        self.video_path = ""  # 视频路径
        self.frame_dleta = 0  # 实际图像序号和标注帧号的增量
        self.id = 0
        self.traj_idx = 0
        
    def bbox_trajectory_analyze(self,anno_files,column_defs):
        """
        brief: 
          针对数据集，就单帧画面出现的样本数目和ID数目做统计
        args:
          anno_files:所有的数据集文件对应的标注文件
          video_dir:视频文件夹路径
          save_dir:统计结果保存目录
          column_defs:VisDrone数据集的标签定义列
        """
        # 1. 获取每个视频文件的所有标注信息
        annos_and_files = {"anno_file":[],"anno":[]}
        if self.dataset_name == "MDMT":
            self.category_mapping, _ = analyze_xml_categories(anno_files)
        for anno_file in anno_files:
            if self.dataset_name == "MDMT":
                anno = load_annotations_xml(anno_file,column_defs,self.category_mapping)
            else:
                anno = load_annotations_txt(anno_file,column_defs)
                
            try:
                get_video_resolution(anno_file,self.dataset_name)
            except Exception as _:
                continue
                  
            anno = remove_nan_entries_cons(anno)
            annos_and_files["anno_file"].append(anno_file)
            annos_and_files["anno"].append(anno)
            
        # 2. 将每组视频文件下每个ID的轨迹间断数、持续时间和运动总位移
        all_traj_num,all_lief_long,all_dist,all_iou,all_gap = [],[],[],[],[]
        for anno_file,anno in zip(annos_and_files["anno_file"],annos_and_files["anno"]):
            video_traj_num,video_lief_long,\
                video_dist,video_iou,video_gaps = \
                self.get_trajectory_statistic(anno_file,anno)
            all_traj_num.extend(video_traj_num)
            all_lief_long.extend(video_lief_long)
            all_dist.extend(video_dist)
            all_iou.extend(video_iou)
            all_gap.extend(video_gaps)
        all_traj_num = np.array(all_traj_num)
        all_lief_long = np.array(all_lief_long)
        all_dist = np.array(all_dist)
        all_iou = np.array(all_iou)
        all_gap = np.array(all_gap)
        # 3. 画统计分布直方图
        visualize_traj_statistic(all_traj_num,all_lief_long,all_dist,all_iou,all_gap,self.save_dir)
        # 4. 对某些视频的某些id进行抽样可视化
        if self.is_show:
            visualize_traj(self.anno_ids,self.save_dir,self.dataset_name,
                           column_defs,self.category_mapping)
  
    def get_trajectory_statistic(self,anno_file,anno):
        """
          brief:
            当前视频文件下的获取所有ID的轨迹统计特性。包括总持续时长，总位移和轨迹断层统计。
          args:
            anno_file:标注文件路径
            anno:当前视频文件的标注条目数组。
        """ 
        # 是否打印异常数据的标志位
        if self.is_show:
            self.video_path,self.frame_dleta = \
            get_frame_dleta(anno,anno_file,self.dataset_name)
        
        video_traj_num,video_lief_long,video_dist,\
            video_iou,video_gaps = [],[],[],[],[]
        # 1. 依次考察该视频下的每个ID
        ID_col = anno["id"]  # 获取ID列
        unique_ID = np.unique(ID_col)  # 该视频文件中所有独立的ID。
        for ID in unique_ID:
            # 为了确保可视化过程的正确性，先将每段视频的第一个ID取出来打印。
            anno_ID = anno[ID_col == ID]
            self.anno_ids.append((anno_file,ID))
            self.id = ID
            # 获取该ID的连续轨迹总数、轨迹的持续总时长、总位移和所有相邻两帧的IOU。
            traj_num,life_long,pos_dist,ID_iou,track_gaps = \
                self.get_id_trajectory(anno_ID)
            if traj_num == 0:
                continue
            video_traj_num.append(traj_num)
            video_lief_long.append(life_long)
            video_dist.append(pos_dist)
            video_iou.extend(ID_iou)
            video_gaps.extend(track_gaps)
        return np.array(video_traj_num),np.array(video_lief_long),\
          np.array(video_dist),np.array(video_iou),np.array(video_gaps)
  
    def get_id_trajectory(self,anno_ID):
        """
        brief:
          计算某id物体的连续轨迹总数、轨迹的持续总时长和总位移
        args:
          anno_ID:本视频内属于该id物体的所有标注
          video_dir:相应的视频文件夹路径
          frame_dleta:实际图像帧序号相对标注帧序号的增量
          save_dir:结果保存文件夹
        return:
          traj_num:连续轨迹数目
          lifelong:该ID出现总时长
          pos_dist:该ID的总位移
          ID_iou:该ID的所有IOU
          track_gaps:该轨迹中间的断裂时间构成的数组
        """
        # 1. 获取该ID出现的总帧数
        frame_ID = np.unique(anno_ID["frame"])
      
        # 2. 根据帧数获取该ID出现帧数断层的数目
        frame_diff = np.diff(frame_ID)
        # *. 可视化 --> 将每一段连续帧的开始和结尾进行打印
        # 3. 将连续轨迹分别分割出来
        trajectories = self.split_continuous_trajectories(anno_ID,frame_diff)
        
        # 4. 针对每一段连续的轨迹依次计算时长和位移长度
        traj_num = len(trajectories) # 该ID的总轨迹数
        if traj_num == 0:
            return 0,0,0.0,np.array([]),np.array([])
        
        if traj_num >= 2:
            track_gaps = self.calculate_track_gaps (trajectories)
        else:
            track_gaps = np.array([])
          
        life_long,pos_dist = 0,0.0
        ID_iou = []
        for traj_index in range(traj_num):
            self.traj_idx = traj_index
            traj_life,traj_dist,traj_iou = self.deal_each_trajectory(
              trajectories[traj_index])  
            life_long = life_long + traj_life
            pos_dist = pos_dist + traj_dist
            ID_iou.extend(traj_iou)
        ID_iou = np.array(ID_iou)
        return traj_num,life_long,pos_dist,ID_iou,track_gaps
      
    def deal_each_trajectory(self,anno_ID):  
        """
        brief:
          针对一条连续的轨迹，获取该轨迹的持续时间和总位移
        args:
          anno_ID:某视频、某ID、某轨迹下的所有标注
          traj_index:轨迹序号
          video_dir:相应的视频文件夹路径
          frame_dleta:实际图像帧序号相对标注帧序号的增量
          save_dir:结果保存文件夹
        return:
          lifelong:该ID出现总时长
          pos_dist:该ID的总位移
          all_iou:该轨迹上的所有IOU
        """
        frame_ID = np.unique(anno_ID["frame"])  # 该轨迹的所有帧数
        lifelong = len(frame_ID)  # 该轨迹的总时长
          
        x = anno_ID["x"]  # 每个帧下边界框的x坐标
        y = anno_ID["y"]  # 每个帧下边界框的y坐标
        diff_x = np.diff(x)  # 相邻两帧边界框的x坐标之差
        diff_y = np.diff(y)  # 相邻两帧边界框的y坐标之差
        pos_dist = np.sum(diff_x * diff_x + diff_y * diff_y)  # 该轨迹的总位移
        
        # 计算每个相邻两帧的iou
        all_iou = self.calculate_all_iou(anno_ID)
            
        return lifelong,pos_dist,all_iou

    def split_continuous_trajectories(self,anno_ID, frame_diff):
        """
        将指定ID的标注数据按照连续轨迹切割成多个子列表
        
        参数:
        anno_ID: 某视频、某ID、某轨迹下的所有标注
        frame_diff: 帧数获取该ID出现帧数断层的数目
        
        返回:
        list: 包含多个连续轨迹的列表，每个轨迹是一个子数组
        """
        id_sorted = np.sort(anno_ID, order='frame')  # 把该ID的轨迹按照出现的帧进行排序
        # 找出不连续的点
        break_indices = np.where(frame_diff > 1)[0] + 1  # +1是因为diff结果比原数组少一个元素
        # 按照不连续的点切割数组
        split_indices = np.concatenate(([0], break_indices, [len(id_sorted)]))
        trajectories = []
        for i in range(len(split_indices)-1):
            start = split_indices[i]
            end = split_indices[i+1]
            if len(id_sorted[start:end]) < 2:
                continue
            trajectories.append(id_sorted[start:end])
        return trajectories
    
    def calculate_track_gaps(self,track_segments):
        """
        计算同一ID多段轨迹之间的间隔时长
        
        参数:
            track_segments: 按时间排序的轨迹段列表
            
        返回:
            相邻轨迹段之间的间隔时长数组
        """
        track_gaps = np.zeros(len(track_segments)-1,dtype=int)  # 初始化间隔时长数组
        for idx in range(1,len(track_segments)):
            last_traj = track_segments[idx-1]  # 上一段轨迹
            last_time = last_traj["frame"][len(last_traj)-1]  # 上一段轨迹的结束时间
            next_traj = track_segments[idx]  # 下一段轨迹
            next_time = next_traj["frame"][0]  # 下一段轨迹的开始时间
            track_gaps[idx-1] = (next_time - last_time)
        return track_gaps
  
##################################### part 6: 计算相邻两帧之间IOU的均值 ##################################
    def calculate_all_iou(self,anno):
        """
          计算一段连续的轨迹上相邻两帧边界框IOU
          
          参数:
          a: 结构化数组，包含以下字段:
              - "x": 所有边界框的左上角横坐标
              - "y": 所有边界框的左上角纵坐标
              - "w": 所有边界框的宽度
              - "h": 所有边界框的高度
          
          返回:
          mean_iou: 相邻两帧边界框IOU的平均值
        """
        # 1. 判断当前的轨迹是否可以计算IOU
        frame_count = len(anno)
        if frame_count < 2:
          # 少于两帧则不够计算IOU
            return 0.0
          
        # 2. 依次计算相邻两帧边界框构成的IOU
        ious = []
        for i in range(frame_count - 1):
            bbox_cur = (anno[i]["x"],anno[i]["y"],anno[i]["w"],anno[i]["h"])
            bbox_next = (anno[i+1]["x"],anno[i+1]["y"],anno[i+1]["w"],anno[i]["h"])
            iou = self.calculate_bbox_iou(bbox_cur,bbox_next)
            if np.isnan(iou):
                continue
            ious.append(iou)
        # 3. 计算IOU均值
        return np.array(ious)
  

    def calculate_bbox_iou(self,bbox_cur,bbox_next):
        """
        brief:
            计算相邻两帧边界框的IOU。
        Args:
            bbox_cur (tuple:x1,y1,w,h): 当前帧的边界框
            bbox_next (tuple:x1,y1,w,h): 下一帧的边界框
        return: 
            iou:这两帧边界框的IOU
        """
        # 1. 把两帧边界框的信息读进来。
        cur_x,cur_y,cur_w,cur_h = bbox_cur
        next_x,next_y,next_w,next_h = bbox_next
        
        # 2. 计算“重叠区域”的(x1,y1,x2,y2)
        inter_x1 = max(cur_x,next_x)
        inter_y1 = max(cur_y,next_y)
        inter_x2 = min(cur_x + cur_w, next_x + next_w)
        inter_y2 = min(cur_y + cur_h, next_y + next_h)
        
        # 判断：有效重叠区域是否存在
        overlap_exist = inter_x2 > (inter_x1 + 1e-7) and inter_y2 > (inter_y1 + 1e-7)
        if not overlap_exist:
            bbox_iou = -1
            return bbox_iou
        
        # 3. 计算重叠区域的(w,h)
        inter_w = inter_x2 - inter_x1
        inter_h = inter_y2 - inter_y1
        
        # 4. 代入IOU的计算公式
        # 4.1 IOU分子————重叠区域面积
        inter_area = inter_w * inter_h  
        # 4.2 IOU分母————两边界框面积之和减去重叠区域面积
        cur_area = cur_w * cur_h
        next_area = next_w * next_h
        union_area = cur_area + next_area - inter_area
        # 4.3 IOU表达式
        bbox_iou = inter_area / union_area
        
        # 4.4 判断0值
        if bbox_iou < 1e-7:
            bbox_iou = 0.0
        
        # 4.5 返回结果
        return bbox_iou
    
    def debug_nan_iou(self,cur_anno,next_anno):
      """
      brief:
          打印iou值为nan或者边界框无交集情况下的相邻边界框作为异常数据处理
      Args:
          cur_anno (结构化数组): 当前时刻的标注信息
          next_anno (结构化数组): 下一时刻的标注信息
      """
      # 1. 构造保存的视频名称
      video_name = get_video_name(self.video_path,self.dataset_name)
            
      # 2. 创建保存结果的文件夹
      save_dir = Path(self.save_dir) / f"visualized_frames" \
        / f"trajectory" / f"no_overlap" / f"{video_name}" / \
          f"{str(self.id)}" / f"{str(self.traj_idx)}"
      save_dir.mkdir(parents=True,exist_ok=True)
      
      # 3. 在图像中打印相邻帧的边界框信息
      frame_num = cur_anno["frame"]
      image_pattern = get_image_name_pattern(self.dataset_name,self.frame_dleta,frame_num)
      img_path = Path(self.video_path) / f"{image_pattern}"
      if not img_path.exists():
          print(f"{img_path} don't exist")
          return
        
      img = cv2.imread(str(img_path))
      cx,cy,cw,ch = cur_anno["x"],cur_anno["y"],cur_anno["w"],cur_anno["h"]
      cv2.rectangle(img,(int(cx),int(cy)),(int(cx+cw),int(cy+ch)),color = (255,255,0))
      nx,ny,nw,nh = next_anno["x"],next_anno["y"],next_anno["w"],next_anno["h"]
      cv2.rectangle(img,(int(nx),int(ny)),(int(nx+nw),int(ny+nh)),color = (0,255,255))
      cv2.imwrite(str(save_dir / f"{image_pattern.split('.')[0]}_iou.jpg"),img)
      visualize_bbox(next_anno,self.video_path,self.frame_dleta,self.dataset_name,save_dir)
    
    
##################################### part 6:针对UAV123进行分析 #####################################
# 针对UAV123数据集的统计实验复现。
def UAV123_analysis():
    # 1. 获取所有的视频-标注组合(针对UAV123数据集)
    dataset_path = "D:/Users/xdche/MOV/MOV_dataset/UAV123/Dataset_UAV123"
    match_samples, mismatch_samples = extract_video_from_uav123(dataset_path)
    # 2. 对帧数与标注数目一一对应的视频组和不对应的视频组分别进行边界框可视化
    sample_UAV123(match_samples,"match")
    sample_UAV123(mismatch_samples,"mismatch")
    
    
######################################### part 7:MOT任务数据集的总体分析框架 #########################################
def MOT_analysis_entry(dataset_input,anno_files,column_defs,dataset_name,args):
    """
    brief:
      MOT任务的数据集分析入口。
      将数据集是否切分为train/val/test，每个子数据集分别进行统计特性的分析等。
    args:
      dataset_input (str): 数据集的总入口路径
      anno_files (list(str)): 所有标注文件的入口路径
      column_defs: 标注的列定义
      dataset_name (str): 数据集名称 
      args:传入的参数列表
    """
    # 1. 构建结果保存文件夹路径
    save_dir = Path(os.path.dirname(dataset_input)) / "result"
    # 2. 考察该数据集是否为可拆分为valid、train和test子集的数据集
    split_dict,split_dirs,has_split = get_dataset_split_paths( \
        dataset_input,anno_files)
    # 2.1 如果可拆，那么每个数据子集分别进行统计特性的分析。
    if has_split:
        for category,sub_files in split_dict.items():
            options = split_dirs[category],sub_files,dataset_name,category,\
                      args.show_process,save_dir,column_defs
            MOT_analysis_steps(options)
    # 2.2 不管可不可拆，都让整个数据集进行统计特性的分析 
    options = [dataset_input],anno_files,dataset_name,"all",\
              args.show_process,save_dir,column_defs
    MOT_analysis_steps(options)
        
        
def MOT_analysis_steps(options):
    """
    brief:
      关于多目标跟踪任务的数据集总体分析流程。
    args:
      options:包含每种输入参数的元组
    """
    # 读取输入参数
    """
        input_path(list):数据集/子集的总入口路径列表。
        anno_files:数据集/子集的所有标注文件。
        dataset_name:数据集名称。
        subset_name:子集名称。例如：train/valid/test/all
        show_progress:是否展示中间可视化结果
        save_dir:保存路径
        column_defs:数据集的定义列
    """
    input_path,anno_files,dataset_name,subset_name,\
    show_progress,save_dir,column_defs = options
    
    # 创建每个子集的保存文件夹
    save_dir = save_dir / f"{subset_name}"
    save_dir.mkdir(parents=True,exist_ok=True)
    
    # 0. 视频长度的统计特性
    frame_analysis(input_path,dataset_name,save_dir)
    
    if not anno_files:
        return
    
    if show_progress:
        visualize_dataset(anno_files,save_dir,column_defs,dataset_name)
    
    # 1. 统计每个视频的样本数目、id数目
    bbox_dense_analyze(anno_files,save_dir,column_defs,dataset_name)
    
    # 2. 计算边界框的整体分布统计特性
    bbox_overall_analyze(anno_files,dataset_name,save_dir,column_defs,show_progress)
    
    # 3. 计算所有id的轨迹间断分布，轨迹持续时间和运动总位移。
    traj_stat = TrajectoryStatistic(dataset_name,save_dir,show_progress)
    traj_stat.bbox_trajectory_analyze(anno_files,column_defs)
    

######################################### part 8:SOT任务数据集的总体分析框架 #########################################
def SOT_analysis_steps(dataset_input,anno_files,dataset_name,args):
    """
    brief:
      MOT任务的数据集分析入口。
      将数据集是否切分为train/val/test，每个子数据集分别进行统计特性的分析等。
    args:
      dataset_input (str): 数据集的总入口路径
      anno_files (list(str)): 所有标注文件的入口路径
      dataset_name (str): 数据集名称 
      args:传入的参数列表
    """  
    # 1. 创建结果保存文件夹路径
    save_dir = Path(os.path.dirname(dataset_input)) / "result"
    
    # 2. 视频总长度统计特性
    frame_analysis([dataset_input],dataset_name,save_dir)
    
    # 3.SOT任务数据集的标注列定义（根据官方文档调整）
    column_defs = [
        ('x', 'f4'),         # 0 - 边界框左上角x
        ('y', 'f4'),         # 1 - 边界框左上角y
        ('w', 'f4'),         # 2 - 边界框宽度
        ('h', 'f4'),         # 3 - 边界框高度
    ]
    
    # 3. 计算边界框的整体分布统计特性
    bbox_overall_analyze(anno_files,dataset_name,save_dir,column_defs,args.show_process)
    
    # 4. 计算所有id的轨迹间断分布，轨迹持续时间和运动总位移。
    traj_stat = TrajectoryStatistic(dataset_name,save_dir,args.show_process)
    traj_stat.bbox_trajectory_analyze(anno_files,column_defs)