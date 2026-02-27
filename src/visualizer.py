# This file is used to visualize the dataset.
import cv2
import random

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
from .loader import \
    get_frame_dleta, \
    get_image_name_pattern, \
    load_annotations_txt,\
    group_by_column,\
    get_bbox, \
    analyze_xml_categories, \
    load_annotations_xml, \
    get_image_files, \
    get_video_name

######################## part 1： 绘制帧数量的分布直方图         #######################
def draw_frame_hist(all_frame_counts,save_dir):
    """
      brief:
          绘制帧数量的分布直方图
      Args:
          all_frame_counts (list): 所有视频帧数量的列表
          save_dir (str): 保存结果的文件夹路径
    """
    # 1. 绘制帧数量的分布直方图
    frame_options = {
        "vars":all_frame_counts,
        "bins_count":100,
        "bins_arange":"log",  # 直方图是按照对数尺度划分的。
        "x_label":"Frame Count (log scale)",
        "y_label":"Video Count Ratio",
        "title":"Frame Count Distribution",
        "save_dir":save_dir,
        "img_name":"frame_hist"
    }
    draw_hist(frame_options)
    print("Frame histogram saved to {}".format(os.path.join(save_dir, "frame_hist.png")))
    
    # 2. 绘制每个视频的帧数条形图
    video_pos = np.arange(len(all_frame_counts)) + 1
    frame_options = {
        "vars": all_frame_counts,
        "bins": video_pos,
        "x_label": "Video Index",
        "y_label": "Frame Count",
        "title": "Frame Count of Each Video",
        "save_dir": save_dir,
        "img_name": "frame_bar"
    }
    draw_bar(frame_options)
    print("Frame bar chart saved to {}".format(os.path.join(save_dir, "frame_bar.png")))
    
    
######################## part 2: 可视化UAV123数据集部分帧的标注框 #######################
def sample_UAV123(samples,match_flag):
    """_summary_
    brief:
        从所有的“视频-标注”组合中随机筛选出若干组进行可视化
    Args:
        samples (list):
        [
            {
                "frame_folder": str,  # 视频帧文件夹路径
                "anno_file": str,  # 标注文件路径
            },
            或者
            {
                "frame_folder": str,  # 视频帧文件夹路径
                "anno_file": list,  # 标注文件路径
            },
            ...
        ]
    """
    # 1. 从所有的视频-标注组合中随机筛选出若干组进行可视化
    select_video_num = 3  # 选择可视化的视频数量
    sample_indexs = random.sample(range(len(samples)),select_video_num)
    visualized_samples = [samples[i] for i in sample_indexs]
    
    # 2. 从每对视频-标注组合中随机筛选出若干帧进行可视化
    select_frame_num = 5  # 选择可视化的帧数量
    for sample in visualized_samples:
        frame_folder = sample["frame_folder"]
        anno_file = sample["anno_file"]
        
        # 3. 获取视频文件夹中的所有帧
        frame_imgs = get_image_files(frame_folder)
        # 4. 从标注文件里获取所有的标注信息构成的numpy数组
        anno = get_annos_from_file(anno_file)
        
        # 5. 创建保存结果的文件夹
        video_name = os.path.basename(frame_folder)
        save_dir = Path(frame_folder).parents[3] / "results" / \
            "visualized_frames" /match_flag/ f"{video_name}"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"{video_name}视频的边界框可视化结果保存在{video_name}文件夹中")
        
        # 6. 选择待可视化的帧的路径
        frame_indexs = random.sample(range(len(frame_imgs)), select_frame_num)
        # 7. 对选定的帧进行可视化
        for frame_index in frame_indexs:
            frame_img = frame_imgs[frame_index]
            frame = cv2.imread(frame_img)
            
            # 8. 在帧上绘制标注框
            if frame_index >= len(anno):
                print(f"{os.path.basename(frame_folder)}\
                    视频的第{frame_index+1}帧没有标注信息!\
                    该视频一共有{len(anno)}组标注条目！")
                continue
            anno_line = anno[frame_index]
            if np.all(anno_line == None):
                print(f"{os.path.basename(frame_folder)}视频的第\
                    {frame_index+1}条标注信息为全None的无效信息!")
                continue
            
            x1,y1,w,h = anno_line
            x2 = x1 + w 
            y2 = y1 + h 
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.imwrite(f"{save_dir}/{frame_index}.jpg",frame)
            
        print(f"视频{video_name}的边界框可视化完成！")
         
def get_annos_from_file(anno_file):
    """
      brief:
        从标注文件中获取所有的标注
      Args:
        anno_file (str): 标注文件路径
        or anno_file (list): 标注文件路径列表
      Returns:
        annos (np.ndarray): 所有的标注信息构成的numpy数组
    """
    if isinstance(anno_file, str):
        assert os.path.exists(anno_file), "标注文件不存在"
        annos = np.loadtxt(anno_file, delimiter=",")
    elif isinstance(anno_file, list):
        annos = np.concatenate([np.loadtxt(f, delimiter=",") for f in anno_file])
    else:
        raise ValueError("标注文件类型错误")
    return annos


def remove_nan_entries(annos):
    """
      brief:
        从标注信息中移除所有全为None的无效条目(仅针对UAV123数据集)
      Args:
        annos (np.ndarray): 所有的标注信息构成的numpy数组
      Returns:
        annos (np.ndarray): 移除无效条目后的标注信息构成的numpy数组
    """
    # 1. 将值为nan的行去掉
    mask = ~np.all(np.isnan(annos), axis=1)
    annos = annos[mask]
    # 2. 将值为0的行去掉
    non_zero_rows = ~np.all(np.abs(annos) < 1e-7,axis=1)
    annos = annos[non_zero_rows]
    return annos


def remove_nan_entries_cons(annos):
    """brief:
        从标注信息中移除所有全为None的无效条目(仅针对UAV123数据集)
      Args:
        annos : 所有的标注信息构成的结构化数组
      Returns:
        annos : 移除无效条目后的标注信息构成的结构化数组
    """
    anno_bbox = np.stack([annos["x"],annos["y"],annos["w"],annos["h"]],axis=1)
    mask = ~np.all(np.isnan(anno_bbox), axis=1)
    annos = annos[mask]
    anno_bbox = anno_bbox[mask]
    non_zero_rows = ~np.all(np.abs(anno_bbox)<1e-7,axis=1)
    annos = annos[non_zero_rows]
    anno_bbox = anno_bbox[non_zero_rows]
    return annos

################################ part 3: 画边界框面积和宽高比的分布直方图 ##############################
def visualize_bbox_wh(area,aspect_ratio,save_dir):
    """
    brief:
      绘制边界框面积和宽高比的分布直方图
    args:
      area: numpy of shape(n,)，所有边界框的面积
      aspect_ratio: numpy of shape(n,)，所有边界框的宽高比
      save_dir: str，结果保存文件夹
    """
    area_options = {
        "vars":area,
        "bins_count":100,
        "bins_arange":"log",  # 直方图是按照均匀尺度划分的。
        "x_label":"Area (log scale)",
        "y_label":"Count Ratio",
        "title":"Bounding Box Areas Distribution",
        "save_dir":save_dir,
        "img_name":"area_hist"
    }
    draw_hist(area_options)
    print("边界框面积分布直方图绘制完成！")
    
    aspect_ratio_options = {
        "vars":aspect_ratio,
        "bins_count":100,
        "bins_arange":"log",  # 直方图是按照对数尺度划分的。
        "x_label":"Aspect Ratio (log scale)",
        "y_label":"Count Ratio",
        "title":"Bounding Box Aspect Distribution",
        "save_dir":save_dir,
        "img_name":"aspect_ratio_hist"
    }
    draw_hist(aspect_ratio_options)
    print("边界框宽高比分布直方图绘制完成！")
    
def draw_hist(options):
    """
    brief:
      分布直方图绘制函数
    args:
      options: dict
      {
        "vars": numpy of shape(n,), 例如：vars为所有边界框的面积或者宽高比构成的数组。
        "bins_count": int, 直方图的柱子数量
        "bins_arange: str, 直方图的横坐标是按照均匀尺度划分的，还是按照对数尺度划分的。
                      目前支持两种输入："uniform"和"log"
        "x_label": str，x轴标签
        "y_label": str，y轴标签
        "title": str，直方图标题
        "save_dir": str，保存直方图文件夹的路径
        "img_name": str，保存直方图的文件名
      }
    """
    # 1. 获取参数
    vars = options["vars"]
    bins_count = options["bins_count"]
    bins_arange = options["bins_arange"]
    x_label = options["x_label"]
    y_label = options["y_label"]
    title = options["title"]
    save_dir = options["save_dir"]
    img_name = options["img_name"]
    
    if not is_valid_array(vars):
        with open(f"{save_dir}/{img_name}.csv", "a") as f:
            f.write("\n")
            f.write("max,min,avg,std,sum\n")
            f.write("{},{},{},{},{}\n".format(\
                np.nan,np.nan,np.nan,np.nan,np.nan
                )
            )  
            f.close()
            print(f"{x_label} is empty.")
        return
    
    # 2. 获取空间中样本的最小值和最大值，用于确定横坐标的范围。
    min_val = np.min(vars)
    max_val = np.max(vars)
    
    # 3. 根据bins_arange确定横坐标的范围。
    is_shift = False  # 是否将数据进行移动了
    if bins_count == 0:
        # 创建一个包含单个区间的bins
        # 策略：以该值为中心，创建一个宽度为1（或根据数据特性调整）的区间
        center = min_val
        bin_width = 1  # 您可以根据数据尺度调整这个宽度，比如 0.1 对于浮点数
    
        # 创建两个边界：中心值 ± 宽度/2
        input_bins = [center - bin_width/2, center + bin_width/2]
    else:
        if bins_arange == "uniform":
            step = (max_val - min_val) / bins_count
            input_bins = round(min_val) + np.arange(int((max_val - min_val) / step + 1)) * step
        elif bins_arange == "log":
            if min_val > 0:
                input_bins = np.logspace(np.log10(min_val), np.log10(max_val+1), bins_count)
            else:
                shift = -min_val + 1e-5  # 确保所有值>0
                shifted_data = vars + shift  # 平移数据
                min_val = np.min(shifted_data)
                max_val = np.max(shifted_data)
                input_bins = np.logspace(np.log10(min_val), np.log10(max_val+1), bins_count)
                is_shift = True
        else:
            raise ValueError("bins_arange参数错误")
    
    # 4. 将落在每个区间的频数改成出现的比例
    if is_shift:
        hist, output_bins = np.histogram(shifted_data, bins=input_bins)
    else:
        hist, output_bins = np.histogram(vars, bins=input_bins)
        
    if np.sum(hist) == len(vars):
        hist = hist / len(vars)
    else:
        hist = hist / np.sum(hist)
    
    # 5. 绘制直方图
    plt.figure(figsize=(10,6))
    
    # 6. 计算每个区间的中点
    bin_width = np.diff(output_bins)  # 区间宽度
    if bins_arange == "log":
        ori_mid_points = np.sqrt(output_bins[1:] * output_bins[:-1])
    elif bins_arange == "uniform":
        ori_mid_points = (output_bins[1:] + output_bins[:-1]) / 2
    else:
        raise ValueError("bins_arange参数错误")
    plt.bar(ori_mid_points, hist, width=bin_width, color='skyblue', edgecolor='black')
    
    if bins_arange == "log":
        plt.xscale('log')
    plt.xlabel(x_label)
    
    plt.ylabel(y_label)
    plt.title(title)
    
    # 5. 保存绘制结果
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    plt.savefig(f"{save_dir}/{img_name}.png")
    plt.close()
    
    # 6. 将参与绘制的变量数组保存到.csv文件中
    if bins_arange == "log":
        ori_mid_points = np.sqrt(output_bins[1:] * output_bins[:-1])
        log_mid_points = (np.log10(output_bins[:-1]) + np.log10(output_bins[1:])) / 2
        df = pd.DataFrame({
            'bin_start': np.round(output_bins[:-1],4),          # 区间起点
            'bin_end': np.round(output_bins[1:],4),             # 区间终点
            'bin_midpoint_log': np.round(log_mid_points,4),  # 对数尺度中点
            'bin_midpoint_original': np.round(ori_mid_points,4),  # 原始尺度中点
            'frequency': np.round(hist,6)                      # 频数
        })
        df.to_csv(f"{save_dir}/{img_name}.csv", index=False)  # 导出到.csv
    else:
        ori_mid_points = (output_bins[1:] + output_bins[:-1]) / 2
        df = pd.DataFrame({
            'bin_start': output_bins[:-1],          # 区间起点
            'bin_end': output_bins[1:],             # 区间终点
            'bin_midpoint_original': np.round(ori_mid_points,1),  # 原始尺度中点
            'frequency': np.round(hist,6),                      # 频数
        })
        df.to_csv(f"{save_dir}/{img_name}.csv", index=False)  # 导出到.csv
        
    # 7. 将计算频率分布直方图的相关变量的统计特性也保存至.csv文件中。
    with open(f"{save_dir}/{img_name}.csv", "a") as f:
        f.write("\n")
        f.write("max,min,avg,std,sum\n")
        f.write("{},{},{},{},{}\n".format(
          np.max(vars),
          np.min(vars), 
          np.mean(vars), 
          np.std(vars),
          np.sum(vars)
          )
        )  
        f.close()


def draw_bar(options):
    """
    brief:
      条形图绘制函数
    args:
      options: dict
      {
        "vars": numpy of shape(n,), 例如：vars为每个视频的总长度构成的数组
        "bins": numpy of shape(n,), 例如：bins为每个视频的相应序号
        "x_label": str，x轴标签
        "y_label": str，y轴标签
        "title": str，直方图标题
        "save_dir": str，保存直方图文件夹的路径
        "img_name": str，保存直方图的文件名
      }
    """
    
    # 1. 获取参数
    vars = options["vars"]
    bins = options["bins"]
    x_label = options["x_label"]
    y_label = options["y_label"]
    title = options["title"]
    save_dir = options["save_dir"]
    img_name = options["img_name"]
    
    # 2. 画图
    plt.figure(figsize=(10,6))
    plt.bar(bins, vars)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    plt.xticks(bins)
    
    plt.savefig(os.path.join(save_dir, f"{img_name}.png"))
    plt.close()
    
    # 3. 数据导出至.csv文件
    df = pd.DataFrame(
        {
            "index":bins,  # 自变量
            "value":vars   # 因变量
        }
    )
    df.to_csv(os.path.join(save_dir, f"{img_name}.csv"),index=False)


def is_valid_array(arr):
    """检查数组是否有效（非空且不包含NaN）"""
    return arr.size > 0 and not np.isnan(arr).any()
    
    
################################# part 4: 画每个ID每帧边界框相对于第一帧的变换分布 ####################################
def visualize_bbox_wh_change(area_ratios,aspect_changes,save_dir):
    """
      brief:
        画每个ID每帧边界框相对于第一帧的变换分布直方图
      args:
        area_ratios:所有ID下的边界框面积比构成的数组
        aspect_changes:所有ID下的边界框长宽高比例的相对比值构成的数组
        save_dir:结果保存文件夹
    """
    area_options = {
        "vars":area_ratios,
        "bins_count":100,
        "bins_arange":"log",  # 直方图是按照均匀尺度划分的。
        "x_label":"Area ratio (log scale)",
        "y_label":"Count Ratio",
        "title":"Bounding Box Areas change Distribution",
        "save_dir":save_dir,
        "img_name":"area_change_hist"
    }
    draw_hist(area_options)
    print("边界框面积变化分布直方图绘制完成！")
    aspect_options = {
        "vars":aspect_changes,
        "bins_count":100,
        "bins_arange":"log",  # 直方图是按照均匀尺度划分的。
        "x_label":"Aspect ratio change (log scale)",
        "y_label":"Count Ratio",
        "title":"Bounding Box Aspect ratio change Distribution",
        "save_dir":save_dir,
        "img_name":"aspect_change_hist"
    }
    draw_hist(aspect_options)
    print("边界框面积变化分布直方图绘制完成！")


def sample_frames(anno,frame,areas,aspect_ratios, n_samples=5):
    """从数据中等间隔抽取n_samples个帧索引"""
    total_frames = len(frame)
    indices = np.linspace(0, total_frames-1, n_samples, dtype=int)
    return {
        'frame_ids': [frame[i] for i in indices],
        'areas': [areas[i] for i in indices],
        'aspect_ratios': [aspect_ratios[i] for i in indices],
        'bbox':[[anno[i]["x"],anno[i]["y"],anno[i]["w"],anno[i]["h"]] for i in indices]
    }
    
    
def visualize_bbox_trend(bbox_id_idx,column_defs,dataset_name,save_dir):
    """
    brief:
      从所有的视频所有的id中挑选部分id，对他们每一帧下的面积和长宽比做折线图。
    args:
      bbox_id_idx (dict):{
        f"{video_path.stem}_{obj_id}":
        {
          "annos":bbox_anno,
          "res_es":video_info["res_es"],
          "video_path":video_info["video_path"],
          "frame_dleta":video_info["frame_dleta"]    
        }      
      }
      column_defs (list): 构造结构数组的列名称
      dataset_name (str): 数据集名称
      save_dir(Path): 结果保存路径
    """
    save_dir = save_dir.parents[0] / "visualized_frames" / "bbox_trend"
    save_dir.mkdir(parents=True,exist_ok=True)
        
    sample_keys = random.sample(list(bbox_id_idx.keys()),k = 3)
            
    for key in sample_keys:
        video_name,_,id = key.rpartition('_')
        save_path = Path(save_dir) / f"{video_name}"
        save_path.mkdir(parents=True,exist_ok=True)
        
        id_info = bbox_id_idx[key]
        
        # 指定id的边界框在整个图像中的面积比
        anno = remove_nan_entries_cons(id_info["annos"])
        bbox_anno = get_bbox(anno,column_defs)
        bbox_anno = remove_nan_entries(bbox_anno)
        areas = bbox_anno[:,2] * bbox_anno[:,3]
        aspect_ratios = bbox_anno[:,2] / np.clip(bbox_anno[:,3], a_min=1e-6, a_max=None)
        
        # 设置图像的横坐标(帧)
        if 'frame' in id_info["annos"].dtype.names:
            frame = anno["frame"]
        else:
            frame = range(len(bbox_anno))
            
        # 从指定的id中随机抽取一些帧，将这些帧上的边界框进行可视化。
        sampled = sample_frames(anno,frame,areas,aspect_ratios)
        
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax[0].plot(frame,areas, label="Area", color = 'orange')
        ax[0].scatter(sampled['frame_ids'], sampled['areas'], c='y', s=50, label='Sampled')
        ax[1].plot(frame,aspect_ratios, label="Aspect Ratio", color = "purple")
        ax[1].scatter(sampled['frame_ids'], sampled['aspect_ratios'], c='r', s=50, label='Sampled')
        
        fig.suptitle(f"BBox Trend for {key}")
        fig.tight_layout()
        fig.savefig(save_path / f"{key}.png")
        plt.close(fig)
        
        for idx in range(len(sampled["bbox"])):
            x1,y1,w,h = sampled["bbox"][idx]
            frame_id = sampled["frame_ids"][idx]
            img_pattern = get_image_name_pattern(dataset_name,bbox_id_idx[key]["frame_dleta"],frame_id)
            img_path = str(os.path.join(bbox_id_idx[key]["video_path"],img_pattern))
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            cv2.rectangle(img,(int(x1),int(y1)),(int(x1+w),int(y1+h)),(0,255,255))
            cv2.imwrite(str(save_path / f"{id}_{img_pattern}"),img)
    
################################### part 5: 可视化VisDrone的标签至图像 ################################# 
    
class AnnoVisualizer:
    def __init__(self,anno_paths,column_defs,dataset_name):
        """
          brief: 
            VisDrone可视化类的初始化
          args:
            anno_paths:抽样的标注文件路径
            column_defs:VisDrone标注列定义
            dataset_name:数据集名称
        """
        anno_path = random.choice(anno_paths)
        if dataset_name == "MDMT":
            category_mapping, _ = analyze_xml_categories(anno_paths)
            self.annotations = load_annotations_xml(anno_path,column_defs,category_mapping)
        else: 
            self.annotations = load_annotations_txt(anno_path,column_defs)
        self.column_defs = column_defs
        self.dataset_name = dataset_name
        self._validate_annotations()
        video_path,frame_dleta = get_frame_dleta(self.annotations,anno_path,dataset_name)
        self.video_path = video_path
        self.frame_dleta = frame_dleta
        
    def _validate_annotations(self):
        """验证标注完整性"""
        if len(self.annotations) == 0:
            raise ValueError("No valid annotations loaded!")
        
        # 检查关键字段范围
        if 'invalid' in self.annotations.dtype.names:
            valid_mask = (
                (self.annotations['w'] > 0) & 
                (self.annotations['h'] > 0) & 
                (self.annotations['invalid'] == 0))
            self.annotations = self.annotations[valid_mask]
        else:
            valid_mask = (
                (self.annotations['w'] > 0) & 
                (self.annotations['h'] > 0))
            self.annotations = self.annotations[valid_mask]
        print(f"After validation: {len(self.annotations)} annotations")
    
    def _generate_color_map(self,col_name:str):
        """为不同ID/类别生成颜色"""
        return {
            id_: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for id_ in np.unique(self.annotations[col_name])
        }  
        
    def sample_to_visualize(self, save_dir, group_type='id',
                             sample_size=3, frames_per_sample=5):
        """
        抽样可视化流程
        :param save_dir: 结果保存路径
        :param group_type: 'id'/'frame'/'class'
        :param sample_size: 抽样组数
        :param frames_per_sample: 每组显示的帧数
        """
        groups = group_by_column(self.annotations,group_type)
        sorted_groups = {k: groups[k] for k in sorted(groups)}
        if group_type == 'class':
            sampled_keys = list(sorted_groups.keys())
        else:
            sampled_keys = random.sample(list(sorted_groups.keys()),
                                        min(sample_size, len(sorted_groups)))
        
        self.visualize_sample(sorted_groups,sampled_keys,group_type,save_dir,frames_per_sample)
        self.export_samples_to_csv(sorted_groups,sampled_keys,save_dir,group_type)
                    
    def visualize_sample(self,groups,sampled_keys,group_type,save_dir,frames_per_sample):
        """
        brief:
          针对抽样的标注依次可视化并保存到文件。
        args:
          groups:按指定的列分组后的标注
          sampled_keys:被抽样的组的序号
          group_type: 'id'/'frame'/'class'
          save_dir: 结果保存路径
          frames_per_sample:每组显示的帧数
        """
        self.color_map = self._generate_color_map(group_type)
        video_name = get_video_name(self.video_path,self.dataset_name)
        for key in sampled_keys:
            print(f"Visualizing {group_type}: {key} in {video_name}")
            group_annos = groups[key]
            # 随机抽取指定数量帧
            if group_type == 'frame':
                self._visualize_frame(key, group_annos,group_type,save_dir)
            else:
                sampled_frames = random.sample(
                    list({a['frame'] for a in group_annos}), 
                    min(frames_per_sample, len(group_annos))
                )
                for frame in sampled_frames:
                    frame_annos = [a for a in group_annos if a['frame'] == frame]
                    self._visualize_frame(frame, frame_annos,group_type,save_dir)

    def _visualize_frame(self, frame_num, annotations,group_type,save_dir):
        """
          brief:
            可视化单帧
          param:
            frame_num:可视化的帧图像序号。
              注：实际拆分的视频文件中的图像帧序号从1开始，而标注文件的初始帧序号从0开始。
            annotations:待打印的标注信息
        """
        img_pattern = get_image_name_pattern(self.dataset_name,self.frame_dleta,frame_num)
        img_file = glob.glob(os.path.join(self.video_path, img_pattern))[0]
        
        if not os.path.exists(img_file):
            print(f"{img_file} don't exists.")
            return
        
        frame = cv2.imread(img_file)
        for anno in annotations:
            vis_dim = anno[group_type]
            x1, y1, w, h = anno['x'],anno['y'],anno['w'],anno['h']
            
            # 绘制边界框
            color = self.color_map[anno[group_type]]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x1+w), int(y1+h)), color, 2)
        
        # # 保存帧
        if group_type == "frame":
            save_path = save_dir / group_type / os.path.basename(self.video_path)
        else:
            save_path = save_dir / group_type / os.path.basename(self.video_path) / str(vis_dim)
        save_path.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(save_path / f"{(frame_num+1):07d}.jpg"),frame)
    
    def export_samples_to_csv(self,groups,sample_keys, save_dir, group_type='id'):
        """导出抽样数据到CSV"""
        for key in list(sample_keys):
            if group_type == "frame":
                save_path = save_dir / group_type / os.path.basename(self.video_path)
            else:
                save_path = save_dir / group_type / os.path.basename(self.video_path) / str(key)
            save_path.mkdir(parents=True,exist_ok=True)
        
            data = groups[key]
            data_save = np.zeros(shape = (len(data),len(self.column_defs))) 
            for row in range(len(data)):
                for col,col_name in enumerate(data[row].dtype.names):
                    data_save[row,col] = data[row][col_name] 
            df = pd.DataFrame(data_save)
            save_path = Path(save_path) / f"{key}.csv"
            df.to_csv(save_path, index=False)
            
            
def visualize_dataset(anno_paths,save_dir,column_defs,dataset_name):
    """
      brief:
        数据集标注规则的理解可视化接口
      args:
        anno_paths
        dataset_path为数据集的总路径
        dataset_name为数据集的名称
    """ 
    try:
        # 1. 初始化VisDrone的可视化类
        dataset_visualize = AnnoVisualizer(anno_paths,column_defs,dataset_name)
        
        # 创建保存文件夹
        save_dir = save_dir / "visualized_frames"
        # 3. 验证某些列的含义
        # 3.1 验证ID列（实验1）
        print("验证ID列分组...")
        dataset_visualize.sample_to_visualize(save_dir,group_type='id')
        
        # 3.2 验证frame列（实验2）
        print("验证frame列分组...")
        dataset_visualize.sample_to_visualize(save_dir,group_type='frame')
        
        # 3.3 验证class列（实验3）
        print("验证class列分组...")
        dataset_visualize.sample_to_visualize(save_dir,group_type='class')
    except Exception as _:
        print(f"抽样的标注文件没有对应的视频信息")
    

######################### part 6: 做样本在数据集中分布密集程度的统计性实验 #########################
def visualize_bbox_dense_video(sample_counts,ID_counts,save_dir):
    """
      brief:
        画出每组视频包含的样本数目、ID数目的条形图
      args:
        sample_counts:每组视频的样本数
        ID_counts:每组视频的ID数
        save_dir:结果保存文件夹
    """
    video_pos = np.arange(len(sample_counts)) + 1
    frame_options = {
        "vars": sample_counts,
        "bins": video_pos,
        "x_label": "Video Index",
        "y_label": "Sample Count",
        "title": "Sample Count of Each Video",
        "save_dir": save_dir,
        "img_name": "sample_bar"
    }
    draw_bar(frame_options)
    print("Sample bar chart saved to {}".format(os.path.join(save_dir, "sample_bar.png")))
    
    video_pos = np.arange(len(ID_counts)) + 1
    frame_options = {
        "vars": ID_counts,
        "bins": video_pos,
        "x_label": "Video Index",
        "y_label": "ID Count",
        "title": "ID Count of Each Video",
        "save_dir": save_dir,
        "img_name": "ID_bar"
    }
    draw_bar(frame_options)
    print("ID bar chart saved to {}".format(os.path.join(save_dir, "ID_bar.png")))
    
    all_sample_counts = np.sum(np.array(sample_counts))
    all_id_counts = np.sum(np.array(ID_counts))
    save_path = os.path.join(save_dir, "sample_ID.csv")
    with open(save_path, "w") as f:
        f.write("\n")
        f.write("sample,ID\n")
        f.write("{},{}\n".format(
          all_sample_counts, 
          all_id_counts, 
          )
        )  
        f.close()
    print("ID sample number statistic saved to {}".format(save_path))


def visualize_bbox_dense_distribution(
    all_sample_counts,save_dir):
    """
    brief:
      画单帧样本数目/ID数目分布的比例直方图
    param:
      all_sample_counts:数据集内所有标注的帧出现的标注数目集合
      save_dir:结果保存文件夹
    """    
    # 1. 绘制样本数目的分布直方图
    save_path = os.path.join(save_dir, "frame_bbox_statistic.csv")
    with open(save_path, "w") as f:
        f.write("\n")
        f.write("len,max,min,avg,sum\n")
        f.write("{},{},{},{},{}\n".format(
          len(all_sample_counts),
          np.max(all_sample_counts), 
          np.min(all_sample_counts), 
          round(np.mean(all_sample_counts)),
          np.sum(all_sample_counts)
          )
        )  
        f.close()
    print("Frame statistics saved to {}".format(save_path))
    
    bins_count = np.max(all_sample_counts) - np.min(all_sample_counts)
    sample_options = {
        "vars":all_sample_counts,
        "bins_count":bins_count,
        "bins_arange":"uniform",  # 直方图是按照对数尺度划分的。
        "x_label":"Sample Count",
        "y_label":"Frame Count Ratio",
        "title":"Sample Count Distribution",
        "save_dir":save_dir,
        "img_name":"sample_hist"
    }
    draw_hist(sample_options)
    print("Sample histogram saved to {}".format(os.path.join(save_dir, "sample_hist.png")))


def visualize_class_freq(all_cls_freq,save_dir):
    """
      brief:绘制不同ID出现频率的条形图
      args:
        all_id_freq:每个ID出现的频率
        save_dir:结果保存路径
    """
    sorted_dict = OrderedDict(sorted(all_cls_freq.items()))
    cls_pos = np.array([int(cls) for cls in sorted_dict.keys()]) + 1
    cls_num = np.array([sorted_dict[cls] for cls in sorted_dict.keys()])
    frame_options = {
        "vars": cls_num,
        "bins": cls_pos,
        "x_label": "class Index",
        "y_label": "Frame Count",
        "title": "Frame Count of Each class",
        "save_dir": save_dir,
        "img_name": "class_freq"
    }
    draw_bar(frame_options)
    print("Class frequency bar chart saved to {}".format(os.path.join(save_dir, "class_freq.png")))
    
############################## part 7:画样本轨迹统计实验的曲线 ##########################
def visualize_traj_statistic(all_traj_num,all_lief_long,all_dist,all_iou,all_gaps,save_dir):
    """
    brief:
      画样本轨迹统计实验的曲线
    Args:
        all_traj_num (numpy): 数据集内所有ID的连续轨迹数目
        all_lief_long (numpy): 数据集内所有ID的持续总时长
        all_dist (numpy): 数据集内所有ID的总位移
        all_iou (numpy): 数据集内所有ID在每个轨迹下相邻两帧IOU构成的数组
        all_gaps (numpy): 数据集内所有同一目标物内相邻轨迹断裂的间隔时间构成的数组
    """
    save_dir = save_dir / "trajectory_statistics"
    save_dir.mkdir(parents=True,exist_ok=True)
    
    # 1. 画所有ID的连续轨迹数目的分布直方图
    bins_count = np.max(all_traj_num) - np.min(all_traj_num)
    traj_options = {
        "vars":all_traj_num,
        "bins_count":bins_count,
        "bins_arange":"uniform",  # 直方图是按照对数尺度划分的。
        "x_label":"Trajectory Count",
        "y_label":"Frame Count Ratio",
        "title":"Trajectory Count Distribution",
        "save_dir":save_dir,
        "img_name":"count_hist"
    }
    draw_hist(traj_options)
    print("所有ID的连续轨迹数目的分布直方图绘制完成")
    
    # 2. 画出所有ID的持续总时长的分布直方图
    bins_count = np.max(all_lief_long) - np.min(all_lief_long)
    traj_options = {
        "vars":all_lief_long,
        "bins_count":bins_count,
        "bins_arange":"log",  # 直方图是按照对数尺度划分的。
        "x_label":"ID Lifelong (log scale)",
        "y_label":"Frame Count Ratio",
        "title":"ID Lifelong Distribution",
        "save_dir":save_dir,
        "img_name":"life_hist"
    }
    draw_hist(traj_options)
    print("所有ID的持续总时长的分布直方图绘制完成")
    
    # 3. 画出所有ID的总位移的分布直方图
    traj_options = {
        "vars":all_dist,
        "bins_count":100,
        "bins_arange":"log",  # 直方图是按照均匀尺度划分的。
        "x_label":"ID Distance (log scale)",
        "y_label":"Count Ratio",
        "title":"ID Distance Distribution",
        "save_dir":save_dir,
        "img_name":"dist_hist"
    }
    draw_hist(traj_options)
    print("所有ID的总位移的分布直方图绘制完成")
    
    # 4. 画每一个样本在相应轨迹时长下的总位移数
    traj_options = {
        "vars": all_dist,
        "bins": all_lief_long,
        "x_label": "ID Lifelong",
        "y_label": "ID DIstance",
        "title": "ID DIstance With Lifelong",
        "save_dir": save_dir,
        "img_name": "dist_life"
    }
    draw_bar(traj_options)
    print("所有ID的总位移随总时长的条形图绘制完成")
    
    # 5. 画出IOU值域内轨迹数目的频率直方图
    traj_options = {
        "vars": all_iou,
        "bins_count": 100,
        "bins_arange":"uniform",
        "x_label": "IOU",
        "y_label": "traj freq",
        "title": "traj freq in each iou bin",
        "save_dir": save_dir,
        "img_name": "iou_hist"
    }
    draw_hist(traj_options)
    print("IOU值域内轨迹数目的频率直方图绘制完成")
    
    # 6. 画出同一目标物的相邻连续轨迹间隔时长分布图
    traj_options = {
        "vars": all_gaps,
        "bins_count": 100,
        "bins_arange":"uniform",
        "x_label": "traj gap",
        "y_label": "traj freq",
        "title": "traj freq in each iou bin",
        "save_dir": save_dir,
        "img_name": "gap_hist"
    }
    draw_hist(traj_options)
    print("相邻连续轨迹的间隔时长频率直方图绘制完成")

def visualize_traj(anno_ids,save_dir,dataset_name,column_defs,
                   category_mapping=None,sample_num = 5):
    """
    brief:
        运动轨迹可视化
    Args:
        anno_id (list): 标注文件-id:该id的标注文件绝对路径和具体的id序号
        save_dir (str): 结果保存路径
        dataset_name (str): 数据集名称
        column_defs(list): 结构化数组的定义列
        category_mapping(dict):类别映射表
    """
    # 创建保存文件夹
    save_dir = Path(save_dir) / f"visualized_frames" / f"trajectory" / f"traj_like"
    save_dir.mkdir(parents=True,exist_ok=True)
    
    # 1. 首先对所有视频的id进行随机抽样
    sample_anno_id = random.sample(anno_ids,k = min(sample_num,len(anno_ids)))
    
    # 2. 可视化每一组ID
    for anno_id in sample_anno_id:
        # 1. 读取标注文件绝对路径和相关ID号
        anno_file = anno_id[0]
        id_idx = anno_id[1]
        
        # 2. 获取该视频中该ID的连续轨迹标注段
        if category_mapping:
            all_anno = load_annotations_xml(anno_file,column_defs,category_mapping)
        else:
            all_anno = load_annotations_txt(anno_file,column_defs)
        all_anno = remove_nan_entries_cons(all_anno)
        
        # 3. 从文件路径获取视频绝对路径
        try:
            video_path,frame_dleta = get_frame_dleta(all_anno,anno_file,dataset_name)
        except Exception as _:
            continue
        
        # 4. 获取该id的相关标注信息
        mask = all_anno["id"] == id_idx
        anno = all_anno[mask]
        del all_anno  
        
        # 5. 根据该ID的所有时间点划分若干连续轨迹
        trajectories = split_continuous_trajectories_vis(anno,\
                            np.diff(np.unique(anno["frame"])))
         
        # 5. 构造保存路径
        video_name = get_video_name(video_path,dataset_name)
        save_path = save_dir / video_name / str(np.unique(anno["id"])[0]) 
        save_path.mkdir(parents=True,exist_ok=True)
        
        for traj_index in range(len(trajectories)):
            # 5.1 构建单轨迹保存路径
            save_path_traj = save_path / f"{str(traj_index)}"
            save_path_traj.mkdir(parents=True,exist_ok=True)
            # 5.2 任选若干帧进行可视化
            visualize_frame(trajectories[traj_index],save_path_traj,video_path,frame_dleta,dataset_name)
            # 5.3 在最后一帧的图像里可视化完整的轨迹
            draw_traj(trajectories[traj_index],video_path,dataset_name,frame_dleta,save_path_traj)


def split_continuous_trajectories_vis(anno_ID, frame_diff):
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
        

def visualize_frame(anno_ID,save_path,video_path,frame_dleta,
                    dataset_name,select_frame_num=10):
        """
        brief:
          在每组轨迹里选取若干帧进行可视化。
        args:
          anno_ID:属于该ID的所有标注条目
          save_path:保存文件夹路径
          video_path:视频文件夹路径
          frame_dleta:实际下载的帧序号相对标注帧序号的增量
          dataset_name:数据集名称
          select_frame_num:可视化帧数目
        """
        # 1.对帧画面进行抽样
        frame_ID = anno_ID["frame"]  # 该ID出现的所有帧
        sample_indexs = random.sample(range(len(frame_ID)),min(select_frame_num-2,len(frame_ID)-2))
        sample_indexs.append(len(frame_ID)-1)
        sample_indexs.append(0)
        
        for index in sample_indexs:
            anno = anno_ID[index]
            x1 = anno["x"]
            y1 = anno["y"]
            w = anno["w"]
            h = anno["h"]
            
            frame_img = get_image_name_pattern(dataset_name,frame_dleta,anno["frame"])
            img_path = Path(video_path) / f"{frame_img}"
            if img_path.exists():
                img = cv2.imread(str(img_path))
            
                cv2.rectangle(img,(int(x1),int(y1)),(int(x1+w),int(y1+h)),color = (0,255,255))
                cv2.imwrite(str(save_path/f"{frame_img.split('.')[0]}_bbox.jpg"),img)
                
                # 补充：如果当前帧不是最后一帧或者第一帧，那么把IOU的可视化也做出来。
                if index == (len(frame_ID) - 1):
                    continue
                
                anno_next = anno_ID[index + 1]
                x1_next = anno_next["x"]
                y1_next = anno_next["y"]
                w_next = anno_next["w"]
                h_next = anno_next["h"]
                
                cv2.rectangle(img,(int(x1_next),int(y1_next)),
                              (int(x1_next+w_next),int(y1_next+h_next)),color = (255,0,255))
                cv2.imwrite(str(save_path/f"{frame_img.split('.')[0]}_iou.jpg"),img)
      
            
def draw_traj(anno,video_path,dataset_name,frame_dleta,save_path):
    """
    brief:
        将ID的轨迹形状在最后一帧图像上进行可视化
    Args:
        anno (结构化数组): 该ID的所有标记条目
        video_path (str): 该ID来自的视频路径
        dataset_name (str): 数据集名称
        frame_dleta (int): 实际下载的帧序号相对标注帧序号的增量
        save_path (str): 保存路径
    """
    # 1. 将原标注中关于边界框的条目抽取出来
    x1,y1,w,h = anno["x"],anno["y"],anno["w"],anno["w"]
    # 2. 由每个边界框的左上角和尺寸计算出边界框的中心点坐标。
    x = x1 + w / 2
    y = y1 + h / 2
    # 3. 将该坐标拼接起来
    center_points = np.stack([x,y],axis=1)
    # 4. 读取出该ID中最后一帧的图像。
    # 这里用for循环的含义是为了防止个别图像读不到的情况。
    draw_idx = 0
    for frame_idx in range(len(anno)-1,-1,-1):
        frame_num = anno["frame"][frame_idx]
        img_pattern = get_image_name_pattern(dataset_name,frame_dleta,frame_num)
        img_path = os.path.join(video_path,img_pattern)
        if os.path.exists(img_path):
            draw_idx = frame_idx
            break
        
    if not os.path.exists(img_path):
        print(f"{img_path} don't exists.")
        return 
    
    img = cv2.imread(img_path)
    # 5. 在该图像上把每个时刻下的点都描出来，绘制运动轨迹。
    for idx in range(draw_idx):
        cur_cx = center_points[idx,0]
        cur_cy = center_points[idx,1]
        next_cx = center_points[idx+1,0]
        next_cy = center_points[idx+1,1]
        cv2.line(img,(int(cur_cx),int(cur_cy)),(int(next_cx),int(next_cy)),color=(0,255,255))
    # 6. 在可视化轨迹的那一帧画面上补画边界框。
    draw_anno = anno[draw_idx]
    x1,y1,w,h = draw_anno["x"],draw_anno["y"],draw_anno["w"],draw_anno["h"]
    cv2.rectangle(img,(int(x1),int(y1)),(int(x1+w),int(y1+h)),color = (0,255,255))
    # 6. 保存结果
    cv2.imwrite(str(save_path / f"traj.jpg"),img)
    # 7. 将该视频下该ID的结果保存到.csv文件中。
    pure_anno = np.stack([anno[name] for name in anno.dtype.names],axis=1)
    pd.DataFrame(pure_anno).to_csv(str(save_path / f"traj.csv"),index=False)
    
    
##################################### iou异常情况可视化 ##################################
def visualize_bbox(anno,video_path,frame_dleta,dataset_name,save_dir):
    """
    brief: 
        在图像中打印边界框
    Args:
        anno (结构化数组): 当前时刻的标注信息
        video_path (Path): 视频文件夹路径
        frame_dleta (int): 实际图片序号相对标注帧号的增量
        dataset_name (str): 数据集名称
        save_dir (Path): 保存文件夹路径
    """
    frame_num = anno["frame"]
    image_pattern = get_image_name_pattern(dataset_name,frame_dleta,frame_num)
    img_path = Path(video_path) / f"{image_pattern}"
    if not img_path.exists():
        print(f"{img_path} don't exist")
        return
    x,y,w,h = anno["x"],anno["y"],anno["w"],anno["h"]
    img = cv2.imread(str(img_path))
    cv2.rectangle(img,(int(x),int(y)),(int(x+w),int(y+h)),color = (255,255,0))
    cv2.imwrite(str(save_dir / f"{image_pattern}"),img)
    return img
    
    