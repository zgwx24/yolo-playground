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
        help="YOLO model to use (optional, for detection boxes). Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x"
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
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="Disable YOLO detection (SAM only, pure segmentation mode)"
    )
    parser.add_argument(
        "--points",
        type=int,
        default=5,
        help="Number of random points for SAM to segment (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Load YOLO only if not disabled
    yolo_model = None
    if not args.no_yolo:
        print(f"Loading YOLO model: {args.model} (optional, for detection boxes)...")
        yolo_model = YOLO(f"{args.model}.pt")
        yolo_model.to(args.device)
    
    print(f"Loading SAM model (main): {args.sam_model}...")
    # SAM checkpoint mapping
    sam_checkpoints = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    }
    
    try:
        # Initialize SAM (MAIN MODEL)
        sam = sam_model_registry[args.sam_model](checkpoint=sam_checkpoints[args.sam_model])
        if args.device == "cuda":
            sam.to(device=torch.device("cuda"))
        else:
            sam.to(device=torch.device("cpu"))
        sam_predictor = SamPredictor(sam)
        print("✓ SAM model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading SAM: {e}")
        return 1
    
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
    print("SAM Mode: Universal segmentation (YOLO optional)")
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
            
            # SET IMAGE FOR SAM (MAIN TASK)
            sam_predictor.set_image(frame)
            
            # SAM Auto-segmentation using random points
            # This finds all salient objects without YOLO
            h, w = frame.shape[:2]
            
            # Generate random points across the image
            np.random.seed(frame_count % 100)  # Pseudo-random for reproducibility
            input_points = np.random.randint(0, min(h, w), size=(args.points, 2))
            input_labels = np.ones(args.points)  # All positive points (foreground)
            
            # Get masks from SAM using random points
            masks, scores, logits = sam_predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True
            )
            
            # Use the best mask (highest IoU score)
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            
            # Get all masks for visualization
            all_objects = []
            
            # Apply SAM masks
            colors_palette = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255),
                (255, 255, 0), (255, 0, 255), (0, 255, 255),
                (128, 0, 0), (0, 128, 0), (0, 0, 128),
            ]
            
            for i, mask in enumerate(masks):
                # Random color for each mask
                color = colors_palette[i % len(colors_palette)]
                
                # Apply mask with transparency
                mask_image = (mask * 255).astype(np.uint8)
                colored_mask = np.zeros_like(frame)
                colored_mask[mask] = color
                display_frame = cv2.addWeighted(
                    display_frame, 0.8, colored_mask, 0.2, 0
                )
                
                # Draw contour
                contours, _ = cv2.findContours(
                    mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(display_frame, contours, -1, color, 2)
                all_objects.append(f"SAM-obj{i+1}")
            
            # OPTIONAL: Add YOLO detection boxes on top
            yolo_info = ""
            if yolo_model is not None:
                results = yolo_model(frame, conf=args.confidence, verbose=False)
                yolo_boxes = results[0].boxes
                yolo_detections = len(yolo_boxes)
                
                if yolo_detections > 0:
                    class_names = results[0].names
                    detected_classes = {}
                    
                    for box in yolo_boxes:
                        cls_id = int(box.cls)
                        cls_name = class_names[cls_id]
                        detected_classes[cls_name] = detected_classes.get(cls_name, 0) + 1
                        
                        # Draw YOLO box (thin line, minimal intrusion)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 165, 255), 1)
                    
                    yolo_info = " + " + ", ".join([
                        f"{name}({count})" for name, count in detected_classes.items()
                    ])
            
            # Display info
            cv2.putText(
                display_frame,
                f"Frame: {frame_count} | SAM: {args.sam_model} | Segments: {len(masks)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            if yolo_info:
                cv2.putText(
                    display_frame,
                    f"YOLO (optional): {yolo_info}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2
                )
            
            # Show frame
            cv2.imshow("SAM Segmentation (YOLO optional)", display_frame)
            
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
