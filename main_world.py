import argparse
import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="YOLO-World open-vocabulary object detection from webcam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main_world.py                                    # Run with default classes
  python main_world.py --classes "person,car,dog,cat"     # Custom classes
  python main_world.py --classes "phone,laptop,cup,book,pen,glasses,wallet,keys"  # More custom classes
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
        default="person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush,pen,pencil,paper,notebook,monitor,desk,lamp,charger,bag,wallet,keys,glasses,headphones,speaker,camera,watch,plant,picture,mirror,curtain,pillow,blanket,towel,soap,shampoo,tissue,trash can,door,window,wall,floor,ceiling",
        help="Comma-separated list of classes to detect. Can be ANY custom classes!"
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
        "--device",
        type=str,
        default="cuda",
        help="Device to use: 'cpu' or 'cuda' (default: cuda)"
    )
    
    args = parser.parse_args()
    
    # Parse classes
    class_list = [cls.strip() for cls in args.classes.split(',')]
    
    print(f"Loading YOLO-World model: {args.model}...")
    print(f"Total classes to detect: {len(class_list)}")
    print(f"Classes: {', '.join(class_list[:10])}{'...' if len(class_list) > 10 else ''}")
    
    model = YOLO(f"{args.model}.pt")
    model.to(args.device)
    
    # Set custom classes for YOLO-World
    model.set_classes(class_list)
    
    print(f"Opening camera device {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: Cannot open camera device {args.camera}")
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
            
            # Visualize results
            annotated_frame = results[0].plot()
            
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
            
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_count} | Detections: {detections} | YOLO-World ({len(class_list)} classes)",
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
