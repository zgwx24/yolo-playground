import argparse
import cv2
import torch
import json
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="YOLO-World open-vocabulary object detection from webcam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main_world.py                                    # Run with default COCO classes
  python main_world.py --classes extended                 # Run with extended preset
  python main_world.py --classes office                   # Run with office preset
  python main_world.py --classes home                     # Run with home preset
  python main_world.py --classes "person,car,dog,cat"     # Custom classes
  python main_world.py --best-only                        # Only show highest confidence per class
  python main_world.py --confidence 0.6 --best-only       # Higher threshold + best only
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m-world",
        help="YOLO-World model to use (default: yolov8m-world). Options: yolov8s-world, yolov8m-world, yolov8l-world, yolov8x-world"
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated list of classes, or preset name: 'default', 'extended', 'office', 'home'"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Confidence threshold (default: 0.5, 50% confidence)"
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
        "--best-only",
        action="store_true",
        help="Only keep the highest confidence detection for each class (default: show all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: 'cpu' or 'cuda' (default: cuda)"
    )
    
    args = parser.parse_args()
    
    # Load classes from file or use default
    class_list = None
    if args.classes is None or args.classes in ['default', 'extended', 'office', 'home']:
        try:
            with open('classes.json', 'r', encoding='utf-8') as f:
                classes_data = json.load(f)
            
            # Support both dict format and direct list format
            if isinstance(classes_data, dict):
                preset = args.classes or 'default'
                class_list = classes_data.get(preset, classes_data.get('default', []))
            elif isinstance(classes_data, list):
                class_list = classes_data
            
            if class_list:
                print(f"Loaded {len(class_list)} classes from classes.json")
            else:
                print("Warning: No classes found in classes.json")
        except FileNotFoundError:
            print("Warning: classes.json not found, using default COCO classes")
    else:
        # Parse custom comma-separated classes
        class_list = [cls.strip() for cls in args.classes.split(',')]
    
    if not class_list:
        print("Error: No classes available")
        return 1
    
    class_list = [cls.strip() for cls in class_list]
    
    print(f"Loading YOLO-World model: {args.model}...")
    print(f"Total classes to detect: {len(class_list)}")
    print(f"Classes: {', '.join(class_list[:10])}{'...' if len(class_list) > 10 else ''}")
    
    try:
        print("Initializing model...")
        model = YOLO(f"{args.model}.pt")
        print("Moving model to device...")
        model.to(args.device)
        
        # Set custom classes for YOLO-World
        print(f"Setting {len(class_list)} classes...")
        model.set_classes(class_list)
        print("Model ready!")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Check GPU memory
    if args.device == "cuda" and torch.cuda.is_available():
        try:
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        except Exception as e:
            print(f"Warning: Could not get GPU info: {e}")
    
    print(f"Opening camera device {args.camera}...")
    try:
        cap = cv2.VideoCapture(args.camera)
        
        if not cap.isOpened():
            print(f"Error: Cannot open camera device {args.camera}")
            return 1
    except Exception as e:
        print(f"Error opening camera: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Set camera to maximum resolution
    resolutions = [
        (4096, 2304),  # 4K DCI
        (3840, 2160),  # 4K UHD
        (2560, 1440),  # 2K
        (1920, 1080),  # Full HD
        (1280, 720),   # HD
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
    
    # Get camera properties
    frame_width = max_width
    frame_height = max_height
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30
    
    print(f"Camera resolution: {frame_width}x{frame_height} @ {fps}fps")
    
    # Setup video writer if saving
    out = None
    if args.save:
        output_file = "output_world.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
        print(f"Saving output to {output_file}")
    
    print("Press 'q' to quit, 's' to save frame")
    print("-" * 50)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame")
                break
            
            frame_count += 1
            
            # Run YOLO-World inference
            results = model(frame, conf=args.confidence, verbose=False)
            
            # Filter to keep only highest confidence detection per class if requested
            if args.best_only:
                boxes = results[0].boxes
                class_names = results[0].names
                
                # Group by class and find highest confidence
                best_boxes = {}
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls)
                    cls_name = class_names[cls_id]
                    conf = float(box.conf)
                    
                    # Keep only if this is the best confidence for this class
                    if cls_name not in best_boxes or conf > best_boxes[cls_name][1]:
                        best_boxes[cls_name] = (i, conf)
                
                # Keep only the indices of best boxes
                best_indices = [idx for idx, _ in best_boxes.values()]
                
                # Create filtered results
                filtered_boxes = [boxes[i] for i in sorted(best_indices)]
                results[0].boxes = filtered_boxes
            
            # Visualize results
            annotated_frame = results[0].plot()
            
            # Make a copy to allow modifications
            annotated_frame = annotated_frame.copy()
            
            # Display info
            detections = len(results[0].boxes)
            class_names = results[0].names
            detected_classes = {}
            
            # Count detected classes
            for box in results[0].boxes:
                cls_id = int(box.cls)
                cls_name = class_names[cls_id]
                detected_classes[cls_name] = detected_classes.get(cls_name, 0) + 1
            
            # Format class info
            class_info = ", ".join([f"{name}({count})" for name, count in detected_classes.items()])
            if not class_info:
                class_info = "No objects"
            
            # Get GPU memory info
            gpu_info = ""
            if args.device == "cuda" and torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1e9  # Convert to GB
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                gpu_percent = (gpu_memory / gpu_total) * 100
                gpu_info = f" | GPU: {gpu_memory:.1f}GB/{gpu_total:.1f}GB ({gpu_percent:.1f}%)"
            
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_count} | Detections: {detections} | YOLO-World ({len(class_list)} classes){gpu_info}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Show detected classes
            cv2.putText(
                annotated_frame,
                f"Detected: {class_info}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            
            # Show frame
            cv2.imshow("YOLO-World Detection", annotated_frame)
            
            # Save frame if enabled
            if out:
                out.write(annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = f"frame_world_{frame_count:05d}.png"
                cv2.imwrite(filename, annotated_frame)
                print(f"Frame saved: {filename}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        print("-" * 50)
        print(f"Total frames processed: {frame_count}")
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
    
    return 0


if __name__ == "__main__":
    exit(main())
