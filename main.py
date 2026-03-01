# This script is used to load the MOV dataset and perform some analysis on it.
import argparse
from ast import arg
import os
from src import *

def parse_args():
    parser = argparse.ArgumentParser(description="A simple script to demonstrate argument parsing.")
    parser.add_argument("--show_process",type=bool,help="The flag to visualize the intermediate results is.",
                        default=False)
    parser.add_argument("--dataset",type=str,help="The dataset to be analyzed.",
                        default="OURS",choices=DATASET_REGISTRY.keys())
    args = parser.parse_args()
    return args


# 实验2：对数据集的样本边界框进行统计分析
def sample_analysis_UAV123(dataset_input,dataset_name,is_show):
    """
    dataset_input: 数据集的绝对路径
    dataset_name: 数据集的名称
    is_show: 是否展示中间结果
    """
    # 1. 获取UAV123所有的标注文件
    UAV123_anno = os.path.join(dataset_input, "anno/UAV123")
    
    aggre_anno_dir = Path(os.path.dirname(dataset_input)) / "new_anno_dir"
    aggre_anno_dir.mkdir(exist_ok=True,parents=True)
    
    anno_files = load_UAV123_annos(UAV123_anno,aggre_anno_dir)
    SOT_analysis_steps(dataset_input,anno_files,dataset_name,is_show)
    
    
# 实验3：针对VisDrone数据集考察单帧样本的密集程度
def visDrone_analysis_entry(dataset_input,dataset_name,is_show):
    """
    dataset_input: 数据集的绝对路径
    dataset_name: 数据集的名称
    is_show: 是否展示中间结果
    """
    # 1. 定义VisDrone的标注列的含义
    column_defs = [
        ('frame', 'i4'),        # 0 - 帧序号
        ('id', 'i4'),      # 1 - 目标ID
        ('x', 'f4'),         # 2 - 边界框左上角x
        ('y', 'f4'),         # 3 - 边界框左上角y
        ('w', 'f4'),         # 4 - 边界框宽度
        ('h', 'f4'),         # 5 - 边界框高度
        ('occlusion', 'f4'),     # 6 - 目标类别
        ('class', 'i4'), # 7 - 遮挡率(0~1)
        ('occlusion_type', 'i4'), # 8 - 遮挡类型
        ('invalid', 'i4')    # 9 - 有效标志
    ]
    # 3. 获取所有的标注文件路径
    anno_files = glob.glob(f"{dataset_input}/**/*.txt",recursive=True)
    
    # 4. 对VisDrone的MOT任务数据集进行考察
    MOT_analysis_entry(dataset_input,anno_files,column_defs,dataset_name,is_show)

    
# 实验4：针对UAVDT数据集进行统计分析
def UAVDT_analysis_entry(dataset_input,dataset_name,is_show):
    """
    dataset_input: 数据集的绝对路径
    dataset_name: 数据集的名称
    is_show: 是否展示中间结果
    """
    # 1. 定义UAVDT标注列的定义
    # DET Groundtruth Format (*_gt_whole.txt)
    column_defs = [
        ('frame', 'i4'),        # 0 - 帧序号
        ('id', 'i4'),        # 1 - 目标ID
        ('x', 'f4'),         # 2 - 边界框左上角x
        ('y', 'f4'),         # 3 - 边界框左上角y
        ('w', 'f4'),         # 4 - 边界框宽度
        ('h', 'f4'),         # 5 - 边界框高度
        ('out-of-view', 'i4'),  # 6 - 落在视野外的遮挡程度
        ('occlusion', 'i4'),  # 7 - 遮挡程度标记
        ('class', 'i4'),     # 8 - 类别标签
    ]
    
    # 2. 标注文件夹的绝对路径
    anno_dir = Path(dataset_input) / f"UAV-benchmark-MOTD_v1.0" / f"GT"
    # 3 获取所有的DET Groundtruth Format的标注文件
    anno_files = glob.glob(os.path.join(str(anno_dir),"*_gt_whole.txt"))
    
    # 5. 对MOT任务的数据集进行统计分析
    MOT_analysis_entry(dataset_input,anno_files,column_defs,dataset_name,is_show)


def MDMT_analysis_entry(dataset_input,dataset_name,is_show):
    """
    dataset_input: 数据集的绝对路径
    dataset_name: 数据集的名称
    is_show: 是否展示中间结果
    """
    # 1. 定义MDMT数据标注列的含义
    column_defs = [
        ('frame', 'i4'),        # 0 - 帧序号
        ('id', 'i4'),        # 1 - 目标ID
        ('x', 'f4'),         # 2 - 边界框左上角x
        ('y', 'f4'),         # 3 - 边界框左上角y
        ('w', 'f4'),         # 4 - 边界框宽度
        ('h', 'f4'),         # 5 - 边界框高度
        ('out-of-view', 'i4'),  # 6 - 落在视野外的遮挡程度
        ('occlusion', 'i4'),  # 7 - 遮挡程度标记
        ('class', 'i4'),     # 8 - 类别标签
    ]
    # 2. 所有标注文件
    anno_files = glob.glob(f"{dataset_input}/**/*.xml",recursive=True)
    # 3. MOT任务接口
    MOT_analysis_entry(dataset_input,anno_files,column_defs,dataset_name,is_show)
    

def MOT20_analysis_entry(dataset_input,dataset_name,is_show):
    """
    dataset_input: 数据集的绝对路径
    dataset_name: 数据集的名称
    is_show: 是否展示中间结果
    """
    # 1. 定义MOT20数据列的定义
    column_defs = [
        ('frame', 'i4'),        # 0 - 帧序号
        ('id', 'i4'),        # 1 - 目标ID
        ('x', 'f4'),         # 2 - 边界框左上角x
        ('y', 'f4'),         # 3 - 边界框左上角y
        ('w', 'f4'),         # 4 - 边界框宽度
        ('h', 'f4'),         # 5 - 边界框高度
        ('confidence', 'i4'),  # 6 - 当前物体是否考虑
        ('class', 'i4'),  # 7 - 类别标签
        ('visibility', 'f4'),     # 8 - 物体可见程度
    ]
    # 2. 所有标注文件
    anno_files = glob.glob(f"{dataset_input}/**/gt.txt",recursive=True)
    # 3. MOT任务接口
    MOT_analysis_entry(dataset_input,anno_files,column_defs,dataset_name,is_show)   
    

def MOT17_analysis_entry(dataset_input,dataset_name,is_show):
    """
    dataset_input: 数据集的绝对路径
    dataset_name: 数据集的名称
    is_show: 是否展示中间结果
    """
    # 1. 定义MOT20数据列的定义
    column_defs = [
        ('frame', 'i4'),        # 0 - 帧序号
        ('id', 'i4'),        # 1 - 目标ID
        ('x', 'f4'),         # 2 - 边界框左上角x
        ('y', 'f4'),         # 3 - 边界框左上角y
        ('w', 'f4'),         # 4 - 边界框宽度
        ('h', 'f4'),         # 5 - 边界框高度
        ('confidence', 'i4'),  # 6 - 当前物体是否考虑
        ('class', 'i4'),  # 7 - 类别标签
        ('visibility', 'f4'),     # 8 - 物体可见程度
    ]
    # 2. 所有标注文件
    anno_files = glob.glob(f"{dataset_input}/**/gt.txt",recursive=True)
    # 3. MOT任务接口
    MOT_analysis_entry(dataset_input,anno_files,column_defs,dataset_name,is_show)
    

def DanceTrack_analysis_entry(dataset_input,dataset_name,is_show):
    """
    dataset_input: 数据集的绝对路径
    dataset_name: 数据集的名称
    is_show: 是否展示中间结果
    """
    # 1. 定义MOT20数据列的定义
    column_defs = [
        ('frame', 'i4'),        # 0 - 帧序号
        ('id', 'i4'),        # 1 - 目标ID
        ('x', 'f4'),         # 2 - 边界框左上角x
        ('y', 'f4'),         # 3 - 边界框左上角y
        ('w', 'f4'),         # 4 - 边界框宽度
        ('h', 'f4'),         # 5 - 边界框高度
        ('confidence', 'i4'),  # 6 - 当前物体是否考虑
        ('class', 'i4'),  # 7 - 类别标签
        ('visibility', 'f4'),     # 8 - 物体可见程度
    ]
    # 2. 所有标注文件
    anno_files = glob.glob(f"{dataset_input}/**/gt.txt",recursive=True)
    # 3. MOT任务接口
    MOT_analysis_entry(dataset_input,anno_files,column_defs,dataset_name,is_show)


def DBT70_analysis_entry(dataset_input,dataset_name,is_show):
    """
    dataset_input: 数据集的绝对路径
    dataset_name: 数据集的名称
    is_show: 是否展示中间结果
    """
    # 1. 标注文件路径
    anno_files = glob.glob(f"{dataset_input}/**/groundtruth_rect.txt",recursive=True)
    # 2. SOT任务接口
    SOT_analysis_steps(dataset_input,anno_files,dataset_name,is_show)
    

def NAT2021_analysis_entry(dataset_input,dataset_name,is_show):
    """
    dataset_input: 数据集的绝对路径
    dataset_name: 数据集的名称
    is_show: 是否展示中间结果
    """
    # 2. 标注文件路径
    anno_files = glob.glob(f"{dataset_input}/**/*.txt",recursive=True)
    anno_files = [f for f in anno_files if 'att' not in f.split(os.sep)]
    # 3. SOT任务接口
    SOT_analysis_steps(dataset_input,anno_files,dataset_name,is_show)
    
    
def UAVTrack112_analysis_step(dataset_input,dataset_name,is_show):
    """
    dataset_input: 数据集的绝对路径
    dataset_name: 数据集的名称
    is_show: 是否展示中间结果
    """
    # 1. 标注文件路径
    anno_entry = os.path.join(dataset_input,"anno")
    anno_files = glob.glob(os.path.join(anno_entry,"*.txt"))
    # 2. SOT任务接口
    SOT_analysis_steps(dataset_input,anno_files,dataset_name,is_show)
    

def OUTS_analysis_step(dataset_input,dataset_name,is_show):
    """
    dataset_input: 数据集的绝对路径
    dataset_name: 数据集的名称
    is_show: 是否展示中间结果
    """
    # 1. 标注文件路径
    anno_files = glob.glob(f"{dataset_input}/**/gt.txt",recursive=True)
    # 2. 我们数据集的数据列定义
    column_defs = [
        ('frame', 'i4'),        # 0 - 帧序号
        ('id', 'i4'),        # 1 - 目标ID
        ('x', 'f4'),         # 2 - 边界框左上角x
        ('y', 'f4'),         # 3 - 边界框左上角y
        ('w', 'f4'),         # 4 - 边界框宽度
        ('h', 'f4'),         # 5 - 边界框高度
        ('confidence', 'i4'),  # 6 - 当前物体是否考虑
        ('class', 'i4'),  # 7 - 类别标签
        ('visibility', 'f4'),     # 8 - 物体可见程度
    ]
    # 3. MOT任务接口
    MOT_analysis_entry(dataset_input,anno_files,column_defs,dataset_name,is_show)
    

def SportsMOT_analysis_entry(dataset_input,dataset_name,is_show):
    """
    dataset_input: 数据集的绝对路径
    dataset_name: 数据集的名称
    is_show: 是否展示中间结果
    """
    # 1. 标注文件路径
    anno_files = glob.glob(f"{dataset_input}/**/*.txt",recursive=True)
    
    # 2. 数据集的数据列定义
    column_defs = [
        ('frame', 'i4'),        # 0 - 帧序号
        ('id', 'i4'),        # 1 - 目标ID
        ('x', 'f4'),         # 2 - 边界框左上角x
        ('y', 'f4'),         # 3 - 边界框左上角y
        ('w', 'f4'),         # 4 - 边界框宽度
        ('h', 'f4'),         # 5 - 边界框高度
        ('confidence', 'i4'),  # 6 - 当前物体是否考虑
        ('class', 'i4'),  # 7 - 类别标签
        ('visibility', 'f4'),     # 8 - 物体可见程度
    ]
    
    # 4. MOT任务接口
    MOT_analysis_entry(dataset_input,anno_files,column_defs,dataset_name,is_show)


DATASET_REGISTRY = {
    "UAV123":{
        "entry": sample_analysis_UAV123,
        "path": f"G:/UAVBenchmark/UAV123/Dataset_UAV123"
    },
    "VisDrone":{
        "entry": visDrone_analysis_entry,
        "path": f"G:/UAVBenchmark/VisDrone2019-MOT/VisDrone2019-MOT"
    },
    "UAVDT":{
        "entry": UAVDT_analysis_entry,
        "path": f"G:/UAVBenchmark/UAVDT/UAV-benchmark"
    },
    "MDMT":{
        "entry": MDMT_analysis_entry,
        "path": f"G:/UAVBenchmark/Multi-Drone-Multi-Object-Detection-and-Tracking/MDMT"
    },
    "MOT20":{
        "entry": MOT20_analysis_entry,
        "path": f"G:/UAVBenchmark/MOT20/MOT20"
    },
    "MOT17":{
        "entry": MOT17_analysis_entry,
        "path": f"G:/UAVBenchmark/MOT17/MOT17"
    },
    "DanceTrack":{
        "entry": DanceTrack_analysis_entry,
        "path": f"G:/UAVBenchmark/DanceTrack/DanceTrack"
    },
    "SportsMOT":{
        "entry": SportsMOT_analysis_entry,
        "path": f"G:/UAVBenchmark/SportsMOT/SportsMOT_example"
    },
    "DBT70":{
        "entry": DBT70_analysis_entry,
        "path": f"G:/UAVBenchmark/DTB70/DTB70"
    },
    "NAT2021":{
        "entry": NAT2021_analysis_entry,
        "path": f"G:/UAVBenchmark/NAT2021/NAT2021/NAT2021-002"
    },
    "UAVTrack112":{
        "entry": UAVTrack112_analysis_step,
        "path": f"G:/UAVBenchmark/UAVTrack112/V4RFlight112"
    },
    "OURS":{
        "entry": OUTS_analysis_step,
        "path": f"D:/BaiduNetdiskDownload/DynUAVI/img_anno/split"
    },
}
    
    
if __name__ == "__main__":
    args = parse_args()
    dataset_name = args.dataset
    print(f"正在分析数据集: {dataset_name}")
    entry_function = DATASET_REGISTRY[dataset_name]["entry"]  # 入口函数
    dataset_input = DATASET_REGISTRY[dataset_name]["path"]  # 数据集路径
    is_show = args.show_process  # 是否展示中间结果
    entry_function(dataset_input,dataset_name,is_show)
    
    