import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor


def main():
    parser = argparse.ArgumentParser(
        description="YOLO + SAM: Object detection + Precise segmentation from webcam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main_sam.py                    # Run with default settings
  python main_sam.py --model yolov8l    # Use larger YOLO model
  python main_sam.py --confidence 0.6   # Set confidence threshold
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m",
        help="YOLO model to use (default: yolov8m). Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output video to file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: 'cpu' or 'cuda' (default: cuda)"
    )
    parser.add_argument(
        "--sam-model",
        type=str,
        default="vit_b",
        help="SAM model size (default: vit_b). Options: vit_b, vit_l, vit_h"
    )
    
    args = parser.parse_args()
    
    print(f"Loading YOLO model: {args.model}...")
    yolo_model = YOLO(f"{args.model}.pt")
    yolo_model.to(args.device)
    
    print(f"Loading SAM model: {args.sam_model}...")
    # SAM checkpoint mapping
    sam_checkpoints = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    }
    
    try:
        # Initialize SAM
        sam = sam_model_registry[args.sam_model](checkpoint=sam_checkpoints[args.sam_model])
        if args.device == "cuda":
            sam.to(device=torch.device("cuda"))
        else:
            sam.to(device=torch.device("cpu"))
        sam_predictor = SamPredictor(sam)
        print("SAM model loaded successfully!")
    except Exception as e:
        print(f"Error loading SAM: {e}")
        print("Falling back to YOLO-only mode...")
        return yolo_only_mode(args, yolo_model)
    
    print(f"Opening camera device {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: Cannot open camera device {args.camera}")
        return 1
    
    # Set camera to maximum resolution
    resolutions = [
        (4096, 2304),
        (3840, 2160),
        (2560, 1440),
        (1920, 1080),
        (1280, 720),
    ]
    
    max_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    max_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    for width, height in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_width == width and actual_height == height:
            max_width = width
            max_height = height
            break
    
    frame_width = max_width
    frame_height = max_height
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30
    
    print(f"Camera resolution: {frame_width}x{frame_height} @ {fps}fps")
    
    # Setup video writer if saving
    out = None
    if args.save:
        output_file = "output_sam.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
        print(f"Saving output to {output_file}")
    
    print("Press 'q' to quit, 's' to save frame")
    print("=" * 60)
    print("YOLO + SAM Mode: High-precision segmentation")
    print("=" * 60)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame")
                break
            
            frame_count += 1
            display_frame = frame.copy()
            
            # YOLO detection
            results = yolo_model(frame, conf=args.confidence, verbose=False)
            yolo_boxes = results[0].boxes
            
            # Get detection info
            detections = len(yolo_boxes)
            class_names = results[0].names
            detected_classes = {}
            
            for box in yolo_boxes:
                cls_id = int(box.cls)
                cls_name = class_names[cls_id]
                detected_classes[cls_name] = detected_classes.get(cls_name, 0) + 1
            
            # SAM segmentation for each detection
            if detections > 0:
                sam_predictor.set_image(frame)
                
                # Convert YOLO boxes to SAM format
                input_boxes = yolo_boxes.xyxy.cpu().numpy()
                
                # Get masks from SAM
                masks, _, _ = sam_predictor.predict(
                    box=input_boxes,
                    multimask_output=False
                )
                
                # Draw masks on frame
                for i, mask in enumerate(masks):
                    # Random color for each mask
                    color = (
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                        np.random.randint(0, 255)
                    )
                    
                    # Apply mask with transparency
                    mask_image = (mask[0] * 255).astype(np.uint8)
                    colored_mask = np.zeros_like(frame)
                    colored_mask[mask[0]] = color
                    display_frame = cv2.addWeighted(
                        display_frame, 0.7, colored_mask, 0.3, 0
                    )
                    
                    # Draw contour
                    contours, _ = cv2.findContours(
                        mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(display_frame, contours, -1, color, 2)
            
            # Format class info
            class_info = ", ".join([
                f"{name}({count})" for name, count in detected_classes.items()
            ])
            if not class_info:
                class_info = "No objects"
            
            # Display info
            cv2.putText(
                display_frame,
                f"Frame: {frame_count} | Detections: {detections} | YOLO: {args.model} + SAM: {args.sam_model}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            cv2.putText(
                display_frame,
                f"Classes: {class_info}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            
            # Show frame
            cv2.imshow("YOLO + SAM Detection", display_frame)
            
            # Save frame if enabled
            if out:
                out.write(display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = f"frame_sam_{frame_count:05d}.png"
                cv2.imwrite(filename, display_frame)
                print(f"Frame saved: {filename}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        print("=" * 60)
        print(f"Total frames processed: {frame_count}")
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
    
    return 0


def yolo_only_mode(args, yolo_model):
    """Fallback to YOLO-only mode if SAM fails to load"""
    print("Running in YOLO-only mode...")
    
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Cannot open camera device {args.camera}")
        return 1
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            results = yolo_model(frame, conf=args.confidence, verbose=False)
            annotated_frame = results[0].plot()
            
            cv2.imshow("YOLO Detection", annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except KeyboardInterrupt:
        pass
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return 0


if __name__ == "__main__":
    exit(main())
