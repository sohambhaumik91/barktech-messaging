
import numpy as np
from utilites import EventEnums, EventObject
from datetime import datetime, timedelta, timezone
import torch
import torch.nn.functional as F
from .metrics import get_intent_score
import asyncio



async def empty_asyncio_queue(q: asyncio.Queue):
    """Empties an asyncio.Queue by retrieving all items."""
    while not q.empty():
        try:
            item = q.get_nowait()
            q.task_done()
        except asyncio.QueueEmpty:
            break

def process_frames_batch_for_dog_presence(
        frame_batch, detector_model,
        pose_model, processor, 
        detection_threhsold = 0.6, batch_size = 16,
        intent_score_threshold = 0.4
    ):
    
    results = detector_model.predict(frame_batch, stream=True)
    bboxes = []
    valid_ix = []
    
    for ix, result in enumerate(results):
        bbox_per_frame = []
        boxes = result.boxes[result.boxes.cls == 16.0]
        if len(boxes) > 0:
            valid_ix.append(ix)
            boxes = boxes[0]
            for box in boxes:
                xywh_yolo = box.xywh[0].cpu().numpy()
                x_center, y_center, w, h = xywh_yolo
                x_tl = x_center - w / 2
                y_tl = y_center - h / 2
                bbox_coco = np.array([x_tl, y_tl, w, h])
                bbox_per_frame.append(bbox_coco)
            bboxes.append(bbox_per_frame)
    #check threshold for dog presence
    percentage_detections = len(bboxes) / batch_size
    if percentage_detections < detection_threhsold:
        #no dog detected
        utc_now = datetime.now(timezone.utc)
        no_dog_detected_event = EventObject(type=EventEnums.NO_DOG_DETECTION, data={}, timestamp=utc_now)
        return no_dog_detected_event
    inputs = processor(np.array(frame_batch)[valid_ix], boxes=bboxes, return_tensors="pt").to(pose_model.device)
    with torch.no_grad():
        outputs = pose_model(**inputs)
    pose_results = processor.post_process_pose_estimation(outputs, boxes=bboxes)
    xy = torch.stack([pose_result[0]["keypoints"] for pose_result in pose_results]).to("cuda")
    scores = torch.stack([pose_result[0]["scores"] for pose_result in pose_results]).to("cuda")
    
    intent_score = get_intent_score(xy, scores, bboxes)
    if intent_score < intent_score_threshold:
        return EventObject(type=EventEnums.DOG_INTEREST_LOW, data={}, timestamp = datetime.now(timezone.utc))    
    
    return EventObject(type=EventEnums.DOG_INTEREST_HIGH, data={}, timestamp = datetime.now(timezone.utc))

    
