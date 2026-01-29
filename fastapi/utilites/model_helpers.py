
import numpy as np
from datetime import datetime, timedelta, timezone
import torch
import torch.nn.functional as F
from .metrics import get_intent_score
import asyncio
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict
import time


class EventEnums(Enum):
    NO_DOG_DETECTION = "NO_DOG_DETECTED"
    DOG_INTEREST_HIGH = "DOG_INTEREST_HIGH"
    DOG_INTEREST_LOW = "DOG_INTEREST_LOW"
    START_GAME = "START_GAME"
    END_GAME = "END_GAME"
    REWARD_DISPENSED = "REWARD_DISPENSED",
    MAX_ROUNDS_REACHED = "MAX_ROUNDS_REACHED"
    MAX_NUM_RETRIES_EXHAUSTED = "MAX_NUM_RETRIES_EXHAUSTED"
    POKE_DONE = "POKE_DONE"
    MAX_POKES_ALLOWED_EXHAUSTED = "MAX_POKES_ALLOWED_EXHAUSTED"
    WELCOME_PLAYER = "WELCOME_PLAYER"
    DOG_FOUND = "DOG_FOUND"
    COOLDOWN_COMPLETE = "COOLDOWN_COMPLETE"
    STOP_STREAM = "STOP_STREAM"
    START_RECEIVING_STREAM = "START_RECEIVING_STREAM"

@dataclass( frozen=True )
class EventObject:
    type: EventEnums
    timestamp: float
    data: Dict
    


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
    start_time = time.perf_counter()
    results = detector_model.predict(frame_batch, imgsz=480, verbose=False)
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
    yolo_done = time.perf_counter()
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
    total_time = time.perf_counter() - start_time
    print(f"⏱️ INFERENCE TOTAL: {total_time:.3f}s | YOLO: {yolo_done-start_time:.3f}s | Pose: {total_time-yolo_done:.3f}s")
    if intent_score < intent_score_threshold:
        return EventObject(type=EventEnums.DOG_INTEREST_LOW, data={}, timestamp = datetime.now(timezone.utc))    
    
    return EventObject(type=EventEnums.DOG_INTEREST_HIGH, data={}, timestamp = datetime.now(timezone.utc))

    
