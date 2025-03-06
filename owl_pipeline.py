from transformers import AutoProcessor, Owlv2ForObjectDetection
import torch
from PIL import Image
import numpy as np
import torch

def filter_highest_confidence_results(boxes, scores, labels):

    best_results = {}

    for box, score, label in zip(boxes, scores, labels):
        label = label.item()  
        if label not in best_results or score > best_results[label]['score']:
            best_results[label] = {'box': box, 'score': score, 'label': label}

    filtered_boxes = torch.stack([result['box'] for result in best_results.values()])
    filtered_scores = torch.tensor([result['score'] for result in best_results.values()])
    filtered_labels = torch.tensor([result['label'] for result in best_results.values()])

    return filtered_boxes, filtered_scores, filtered_labels


def label_image(image_path,
                text_prompt,
                device):
    
    model = Owlv2ForObjectDetection.from_pretrained("./scenic/owlv2-large-patch14-ensemble").to(device)
    processor = AutoProcessor.from_pretrained("./digital_twin/scenic/owlv2-large-patch14-ensemble")
    
    image = Image.open(image_path)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    all_boxes = []
    all_scores = []
    all_labels = []

    for idx, text_query in enumerate(text_prompt):
        inputs = processor(text=[text_query], images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        target_sizes = torch.Tensor([image.size[::-1]]).to(device)
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
        
        boxes = results[0]["boxes"]
        scores = results[0]["scores"]
        labels = results[0]["labels"]
        mapped_labels = torch.full_like(labels, idx)  
        
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(mapped_labels)

    all_boxes = torch.cat(all_boxes, dim=0) 
    all_scores = torch.cat(all_scores, dim=0) 
    all_labels = torch.cat(all_labels, dim=0) 

    
    boxes, scores, labels = all_boxes, all_scores, all_labels
    return boxes, scores, labels

def find_label(boxes, scores, labels, mask_path, text_queries):
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask) / 255.0
    mask = (mask > 0.5).astype(np.float32)  
    mask = torch.tensor(mask, dtype=torch.float32)
    
    max_iou = 0
    max_overlap_box = None
    max_overlap_score = None
    max_overlap_label = None

    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        overlap_area = mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])].sum()
        
        box_area = (int(box[2]) - int(box[0])) * (int(box[3]) - int(box[1]))
        
        mask_area = mask.sum()
        
        iou = overlap_area / (box_area + mask_area - overlap_area)
        
        if iou > max_iou:
            max_iou = iou
            max_overlap_box = box
            max_overlap_score = score
            max_overlap_label = label

    if max_overlap_box is not None:
        return text_queries[max_overlap_label]
    else:
        return 'Unknown'
