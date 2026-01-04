#!/usr/bin/env python3
"""
Convert Open Images Dataset v6 class descriptions CSV to JSON
"""
import csv
import json

def convert_csv_to_json():
    classes = []
    
    print("Reading oidv6-class-descriptions.csv...")
    with open('oidv6-class-descriptions.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 使用DisplayName作为类别名
            display_name = row['DisplayName'].lower().strip()
            if display_name:
                classes.append(display_name)
    
    print(f"Loaded {len(classes)} classes")
    
    # 保存为JSON
    output_data = {
        "default": classes,
        "description": f"Open Images Dataset v6 - {len(classes)} classes"
    }
    
    with open('classes.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to classes.json")
    print(f"Sample classes: {classes[:10]}")

if __name__ == "__main__":
    convert_csv_to_json()
