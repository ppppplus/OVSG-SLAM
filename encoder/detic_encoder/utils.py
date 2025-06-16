import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def find_enclosing_rectangle(box1, box2):
    """
    Find the bounding rectangle of two bounding boxes.
    
    Args:
        box1: First bounding box (x1, y1, x2, y2)
        box2: Second bounding box (x1, y1, x2, y2)
    
    Returns:
        tuple: Bounding rectangle (x1, y1, x2, y2)
    """
    print(box1, box2)
    x1 = min(box1[0], box2[0])  # Top-left x
    y1 = min(box1[1], box2[1])  # Top-left y
    x2 = max(box1[2], box2[2])  # Bottom-right x
    y2 = max(box1[3], box2[3])  # Bottom-right y

    return (x1, y1, x2, y2)

def task_foreground_extract(seg_img, class_name, pred_boxes, threshold=0.6):
    # seg_img: Segmentation image with annotation IDs, class_name: Class name, pred_boxes: Predicted bounding box coordinates
    if "table" in class_name:
        table_index = np.where(class_name=="table")
    elif "dining_table" in class_name:
        table_index = np.where(class_name=="dining_table")
    else:
        return np.ones(seg_img.shape, dtype=bool)   # true
    table_box = pred_boxes[table_index][0]

    # 提取方框的边界
    x_min, y_min, x_max, y_max = table_box
    # 确保坐标在图像范围内
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, seg_img.shape[1])
    y_max = min(y_max, seg_img.shape[0])
    table_box = (x_min, y_min, x_max, y_max)
    box_area = (x_max - x_min) * (y_max - y_min)
    # 提取方框区域
    roi = seg_img[y_min:y_max, x_min:x_max]
    # 获取所有唯一的 mask 代号
    unique_masks = np.unique(roi)
    masks_above_threshold = np.zeros_like(seg_img)
    box_above = table_box
    # box_areas = []
    # total_mask_areas = []
    # intersections = []
    # overlaps = []
    print(unique_masks)
    for mask in unique_masks:
        # 创建二值掩码，用于计算与方框的交集
        mask_area = (seg_img == mask)
        mask_in_box = mask_area[y_min:y_max, x_min:x_max]
        # 计算交集
        intersection = np.sum(mask_in_box)
        # 计算 mask 的总面积
        total_mask_area = np.sum(mask_area)
        # 计算重合度
        overlap = intersection / total_mask_area
        # 判断重合度是否大于阈值
        if overlap > threshold:
            masks_above_threshold += mask_area
            
            print(pred_boxes[mask-1])
            box_above = find_enclosing_rectangle(box_above, pred_boxes[mask-1])

    res_mask = (masks_above_threshold > 0)
    res_mask = dilate_mask_boundary_opencv(res_mask, 3)
    return res_mask, box_above

def dilate_mask_boundary_opencv(mask, dilation_size=1):
    """
    使用OpenCV对布尔mask的边界进行扩展（膨胀）。
    
    参数:
    mask -- 原始布尔mask
    dilation_size -- 扩展的尺寸，默认为1
    
    返回:
    扩展后的mask
    """
    # 创建一个结构元素
    kernel = np.ones((dilation_size*2+1, dilation_size*2+1), dtype=np.uint8)
    
    # 将mask转换为uint8类型
    mask_uint8 = mask.astype(np.uint8)
    
    # 执行膨胀操作
    dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)
    
    # 将结果转换回布尔类型
    dilated_mask = dilated_mask.astype(bool)
    
    return dilated_mask

if __name__ == "__main__":
    for i in range(50):
        fpath = f"/home/langsplat_ws/myfeature-3dgs/feature-3dgs/datasets/isaac_50/detic_embeddings/{i:04d}-color_dict.pkl"
        with open(fpath, 'rb') as f:
            data_dict = pickle.load(f)
        seg_img = data_dict["seg_image"]

        class_name = data_dict["class_names"]
        # class_indices = data_dict["class_indices"]
        pred_boxes = data_dict["pred_boxes"]
        # # print(class_name)
        # if "table" in class_name:
        #     table_index = np.where(class_name=="table")
        # elif "dining_table" in class_name:
        #     table_index = np.where(class_name=="dining_table")
        # else:
        #     continue

        # table_box = pred_boxes[table_index]
        input_mask = task_foreground_extract(seg_img, class_name, pred_boxes)
        # input_mask = dilate_mask_boundary_opencv(input_mask, 3)
        
        # plt.imshow(input_mask)
        # plt.show()
        print(i, "mask get")