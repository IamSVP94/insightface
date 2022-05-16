import torch
from collections import Counter, defaultdict


def count_all_needed_indicators(iou_results, correct_boxes, detected_boxes):
    not_find_face = len(correct_boxes) - len(iou_results.values())
    fp = len(detected_boxes) - len(iou_results.values())
    iou_lst = [iou_results[item][1] for item in iou_results]
    return not_find_face, fp, iou_lst


def bb_intersection_over_union(box_a, box_b):
    """
    Parameters:
        box_a(list): first bounding box
        box_b(list): second bounding box
    Returns:
        iou(float): intersection over union value
    """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    return iou


def find_best_iou_for_many(dd, iou_results):
    """
    This is the function we use if we have
    two or more correct boxes per one detected box.
    We remove from the dictionary(iou_results)
    the box that has less iou (so, we attribute this box to a false positive).
    """
    for k, v in dd.items():
        max_iou_value = 0
        values = list(v)
        for i, value in enumerate(values):
            if iou_results[value][1] > max_iou_value:
                if i != 0:
                    del iou_results[values[i - 1]]
            else:
                del iou_results[values[i]]
                max_iou_value = iou_results[value][1]
    return iou_results


def find_iou_for_all_boxes(correct_boxes, detected_boxes):
    """
    Parameters:
         correct_boxes(list of list): labeled bounding boxes
         detected_boxes(list of list): bounding boxes that return NN
    Returns:
        not_find_face(int): number of missing faces
        fp(int): the number of objects(faces) found where there are none
        iou_lst(int): intersection over Union all found faces in the image
    """
    iou_results = {}
    for true_box in correct_boxes:
        for detected_box in detected_boxes:
            iou = bb_intersection_over_union(true_box, detected_box)
            if str(true_box) in iou_results:
                if iou_results[str(true_box)][1] < iou:
                    iou_results[str(true_box)] = [detected_box, iou]
            else:
                iou_results[str(true_box)] = [detected_box, iou]

        # remove false positive result
        if iou_results:
            last_element_in_dct = iou_results[list(iou_results)[-1]]
            if last_element_in_dct[1] == 0.0:
                del iou_results[list(iou_results)[-1]]

    dd = defaultdict(set)

    for key, value in iou_results.items():
        dd[str(value[0])].add(key)
    dd = {k: v for k, v in dd.items() if len(v) > 1}
    if dd:
        iou_results = find_best_iou_for_many(dd, iou_results)
    not_find_face, fp, iou_lst = count_all_needed_indicators(iou_results, correct_boxes, detected_boxes)
    fp_or_fn_faces = [0 for i in range(not_find_face + fp)]  # add 0 if FP or FN faces
    iou_lst.extend(fp_or_fn_faces)
    return not_find_face, fp, iou_lst


def intersection_over_union(boxes_preds: list, boxes_labels: list, box_format: str = "midpoint") -> torch.Tensor:
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (list): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (list): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """
    if boxes_preds == boxes_labels and boxes_labels == []:
        return torch.Tensor([1])
    elif boxes_preds != boxes_labels and (boxes_preds == [] or boxes_labels == []):
        return torch.Tensor([0])

    # Slicing idx:idx+1 in order to keep tensor dimensionality
    boxes_preds = torch.Tensor(boxes_preds)
    boxes_labels = torch.Tensor(boxes_labels)

    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would have (N, S, S, 4) in shape
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_y1 = boxes_preds[..., 2:3] - boxes_preds[..., 4:5] / 2
        box1_x2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box1_y2 = boxes_preds[..., 2:3] + boxes_preds[..., 4:5] / 2

        box2_x1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_y1 = boxes_labels[..., 2:3] - boxes_labels[..., 4:5] / 2
        box2_x2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
        box2_y2 = boxes_labels[..., 2:3] + boxes_labels[..., 4:5] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 1:2]
        box1_y1 = boxes_preds[..., 2:3]
        box1_x2 = boxes_preds[..., 3:4]
        box1_y2 = boxes_labels[..., 0:1]
        box2_x1 = boxes_labels[..., 1:2]
        box2_y1 = boxes_labels[..., 2:3]
        box2_x2 = boxes_labels[..., 3:4]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    final_iou = intersection / (box1_area + box2_area - intersection + 1e-6)
    return final_iou


def mean_average_precision(pred_boxes: list, true_boxes: list,
                           iou_threshold=0.5,
                           box_format: str = "midpoint",
                           num_classes: int = 20) -> float:
    """
    Calculates mean average precision
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """
    if pred_boxes == true_boxes and true_boxes == []:
        return 1.0
    elif pred_boxes != true_boxes and (pred_boxes == [] or true_boxes == []):
        return 0.0

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[0] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[0] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])  # +

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():  # - ???
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                # iou = intersection_over_union(detection, gt, box_format=box_format)
                _, _, iou = find_iou_for_all_boxes([detection], [gt])
                if iou[-1] > best_iou:
                    best_iou = iou[-1]
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

        precisions = torch.cat((torch.tensor([1.0]), precisions))
        recalls = torch.cat((torch.tensor([0.0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    final_map = sum(average_precisions) / len(average_precisions)
    return final_map


def get_GT_bbox(txt_path):
    with open(txt_path, 'r') as txt:
        lines = txt.readlines()
        GT = []
        for line in lines:
            line = line.strip('\n').split(' ')
            cl, x, y, w, h = line
            GT.append([int(cl), float(x), float(y), float(w), float(h)])
    return GT
