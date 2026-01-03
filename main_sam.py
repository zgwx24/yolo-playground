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
        default="yolov8n",
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
    parser.add_argument(
        "--fps-limit",
        type=int,
        default=0,
        help="Limit FPS for better performance (0 = no limit)"
    )
    parser.add_argument(
        "--sam-interval",
        type=int,
        default=3,
        help="Process SAM every N frames (default: 3, higher = faster but less frequent updates)"
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=640,
        help="Maximum resolution for processing (default: 640, lower = faster)"
    )
    parser.add_argument(
        "--use-yolo-for-sam",
        action="store_true",
        help="Only run SAM on YOLO-detected regions (much faster, requires YOLO)"
    )
    
    args = parser.parse_args()
    
    # Load YOLO only if not disabled
    yolo_model = None
    if not args.no_yolo:
        print(f"Loading YOLO model: {args.model} (optional, for detection boxes)...")
        yolo_model = YOLO(f"{args.model}.pt")
        yolo_model.to(args.device)
    
    print(f"Loading SAM model (main): {args.sam_model}...")
    
    # Download SAM checkpoint if not exists
    import os
    from pathlib import Path
    
    sam_checkpoint_map = {
        "vit_b": "sam_vit_b_01ec64.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_h": "sam_vit_h_4b8939.pth",
    }
    
    checkpoint_name = sam_checkpoint_map[args.sam_model]
    checkpoint_dir = Path.home() / ".cache" / "segment_anything"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / checkpoint_name
    
    # Download if not exists
    if not checkpoint_path.exists():
        import urllib.request
        checkpoint_urls = {
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        }
        url = checkpoint_urls[args.sam_model]
        print(f"Downloading {args.sam_model} checkpoint (~{[375, 1200, 2500][['vit_b', 'vit_l', 'vit_h'].index(args.sam_model)]}MB)...")
        urllib.request.urlretrieve(url, checkpoint_path)
        print(f"✓ Downloaded to {checkpoint_path}")
    
    try:
        # Initialize SAM (MAIN MODEL)
        sam = sam_model_registry[args.sam_model](checkpoint=str(checkpoint_path))
        if args.device == "cuda":
            sam.to(device=torch.device("cuda"))
            # Note: FP16 disabled for stability - SAM has compatibility issues with half precision
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
    
    # Set camera to optimized resolution for GTX 1060
    # Lower resolution = faster processing
    target_resolutions = [
        (640, 480),   # VGA - best for GTX 1060
        (800, 600),
        (1280, 720),  # HD
        (1920, 1080), # Full HD (may be too slow)
    ]
    
    # Find best resolution that doesn't exceed max_resolution
    frame_width = 640
    frame_height = 480
    
    for width, height in target_resolutions:
        if width <= args.max_resolution:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_width >= width * 0.9:  # Allow some tolerance
                frame_width = actual_width
                frame_height = actual_height
                break
    
    # Force to max_resolution if needed
    if frame_width > args.max_resolution:
        scale = args.max_resolution / frame_width
        frame_width = args.max_resolution
        frame_height = int(frame_height * scale)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
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
    print(f"Optimized for GTX 1060: {frame_width}x{frame_height}, SAM every {args.sam_interval} frames")
    if args.use_yolo_for_sam:
        print("YOLO-guided SAM: Only processing detected regions (faster)")
    print("=" * 60)
    
    frame_count = 0
    import time
    start_time = time.time()
    frame_times = []
    
    # Cache for SAM processing
    last_sam_mask = None
    last_sam_frame = None
    
    # Enable memory efficient attention if available
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    
    try:
        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame")
                break
            
            frame_count += 1
            display_frame = frame.copy()
            
            # OPTIONAL: Add YOLO detection boxes first (faster, can guide SAM)
            yolo_info = ""
            yolo_boxes_list = []
            if yolo_model is not None:
                results = yolo_model(frame, conf=args.confidence, verbose=False, imgsz=640)
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
                        yolo_boxes_list.append((x1, y1, x2, y2))
                    
                    yolo_info = " + " + ", ".join([
                        f"{name}({count})" for name, count in detected_classes.items()
                    ])
            
            # SAM processing - only every N frames to save GPU memory
            should_process_sam = (frame_count % args.sam_interval == 0) or (last_sam_mask is None)
            
            if should_process_sam:
                # Clear GPU cache before SAM processing
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Set memory fraction to prevent OOM
                    torch.cuda.set_per_process_memory_fraction(0.9)
                
                # Resize frame for SAM to reduce memory usage
                h_orig, w_orig = frame.shape[:2]
                if max(h_orig, w_orig) > args.max_resolution:
                    scale = args.max_resolution / max(h_orig, w_orig)
                    new_w = int(w_orig * scale)
                    new_h = int(h_orig * scale)
                    frame_sam = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                else:
                    frame_sam = frame
                    new_h, new_w = h_orig, w_orig
                
                # SET IMAGE FOR SAM (only when needed)
                sam_predictor.set_image(frame_sam)
                
                # SAM Auto-segmentation using random points or YOLO boxes
                if args.use_yolo_for_sam and yolo_boxes_list:
                    # Use YOLO boxes as prompts (much more efficient)
                    all_masks = []
                    for x1, y1, x2, y2 in yolo_boxes_list:
                        # Convert to SAM scale
                        x1_sam = int(x1 * new_w / w_orig)
                        y1_sam = int(y1 * new_h / h_orig)
                        x2_sam = int(x2 * new_w / w_orig)
                        y2_sam = int(y2 * new_h / h_orig)
                        
                        # Use center point of box
                        center_x = (x1_sam + x2_sam) // 2
                        center_y = (y1_sam + y2_sam) // 2
                        
                        input_points = np.array([[center_x, center_y]])
                        input_labels = np.array([1])
                        
                        masks, scores, logits = sam_predictor.predict(
                            point_coords=input_points,
                            point_labels=input_labels,
                            multimask_output=False
                        )
                        all_masks.append(masks[0])
                    
                    # Combine all masks
                    if all_masks:
                        combined_mask = np.zeros((new_h, new_w), dtype=bool)
                        for mask in all_masks:
                            combined_mask = combined_mask | mask
                        best_mask = combined_mask
                    else:
                        best_mask = np.zeros((new_h, new_w), dtype=bool)
                else:
                    # Random points approach (original method)
                    input_points = np.random.randint(0, min(new_h, new_w), size=(args.points, 2))
                    input_labels = np.ones(args.points)
                    
                    masks, scores, logits = sam_predictor.predict(
                        point_coords=input_points,
                        point_labels=input_labels,
                        multimask_output=False
                    )
                    best_mask = masks[0]
                
                # Resize mask back to original size if needed
                if (new_h != h_orig) or (new_w != w_orig):
                    best_mask = cv2.resize(
                        best_mask.astype(np.uint8), 
                        (w_orig, h_orig), 
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                
                last_sam_mask = best_mask
                last_sam_frame = frame_count
            else:
                # Reuse last mask
                best_mask = last_sam_mask
            
            # Apply mask visualization
            if best_mask is not None:
                color = (0, 255, 0)
                mask_image = (best_mask * 255).astype(np.uint8)
                colored_mask = np.zeros_like(frame)
                colored_mask[best_mask] = color
                display_frame = cv2.addWeighted(
                    display_frame, 0.85, colored_mask, 0.15, 0
                )
                
                # Draw contour (simplified)
                contours, _ = cv2.findContours(
                    mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(display_frame, contours, -1, color, 1)
            
            # Calculate FPS
            frame_end = time.time()
            frame_time = frame_end - frame_start
            frame_times.append(frame_time)
            if len(frame_times) > 30:
                frame_times.pop(0)
            
            avg_fps = len(frame_times) / sum(frame_times) if frame_times else 0
            
            # Display info
            sam_status = f"SAM: {args.sam_model} (every {args.sam_interval} frames)"
            if last_sam_frame:
                frames_since_sam = frame_count - last_sam_frame
                sam_status += f" [{frames_since_sam} ago]"
            
            cv2.putText(
                display_frame,
                f"Frame: {frame_count} | FPS: {avg_fps:.1f} | {sam_status}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
            
            if yolo_info:
                cv2.putText(
                    display_frame,
                    f"YOLO: {yolo_info}",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    1
                )
            
            # Show frame
            cv2.imshow("SAM Segmentation", display_frame)
            
            # Save frame if enabled
            if out:
                out.write(display_frame)
            
            # FPS limiting if specified
            if args.fps_limit > 0:
                sleep_time = max(0, (1.0 / args.fps_limit) - frame_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
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
