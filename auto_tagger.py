import os
import re
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types

# Define the expected JSON structure for Gemini
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {
            "type": "string",
            "description": "A simple, meaningful description of the image in 10 words or less. Do not use special characters."
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "10 to 20 tags describing the format, subjects, and style of the image."
        }
    },
    "required": ["description", "tags"]
}

def get_year_from_metadata(filepath):
    """Safely extracts the original creation year from EXIF metadata using Pillow."""
    try:
        with Image.open(filepath) as img:
            exif = img.getexif()
            if exif:
                # Tag 36867 is DateTimeOriginal
                date_str = exif.get(36867) 
                if date_str:
                    return date_str.split(":")[0]
    except Exception:
        pass
    return None

def inject_metadata_safely(filepath, tags):
    """Uses ExifTool to inject metadata without altering the image bitstream."""
    tags_str = ", ".join(tags)
    
    command = [
        "exiftool",
        "-overwrite_original",    
        f"-sep", ", ",            
        f"-XMP:Subject+={tags_str}", 
        f"-IPTC:Keywords+={tags_str}",
        str(filepath)
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        print("[!] ERROR: ExifTool is not installed or not in PATH.")
        return False
    except subprocess.CalledProcessError:
        print(f"[!] ERROR: Failed to write metadata to {filepath.name}")
        return False

def sanitize_filename(name):
    """Removes characters that are illegal in Windows/Mac file paths."""
    safe_name = re.sub(r'[\\/*?:"<>|\x00-\x1f]', "", name)
    return safe_name.strip()

def process_image(client, filepath, args):
    print(f"\nProcessing: {filepath.name}")
    
    # 1. Upload and Analyze with Gemini
    print(f"  [*] Analyzing with {args.model}...")
    try:
        uploaded_file = client.files.upload(file=str(filepath))
        
        response = client.models.generate_content(
            model=args.model,
            contents=[uploaded_file, args.prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
                temperature=0.4 
            ),
        )
        
        client.files.delete(name=uploaded_file.name)
        
        result = json.loads(response.text)
        description = result.get("description", "Unknown Image")
        tags = result.get("tags", [])
        
    except Exception as e:
        print(f"  [!] Gemini API Error: {e}")
        return

    # 2. Get Year from Metadata
    year = get_year_from_metadata(filepath)
    
    # 3. Construct New Filename & Path
    clean_desc = sanitize_filename(description)
    new_filename = f"{clean_desc} ({year}){filepath.suffix}" if year else f"{clean_desc}{filepath.suffix}"
    
    # Determine target directory
    target_dir = Path(args.output_dir) if args.output_dir else filepath.parent
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        
    new_filepath = target_dir / new_filename

    # Handle filename collisions
    counter = 1
    while new_filepath.exists() and new_filepath.resolve() != filepath.resolve():
        collision_name = f"{clean_desc} ({year}) [{counter}]{filepath.suffix}" if year else f"{clean_desc} [{counter}]{filepath.suffix}"
        new_filepath = target_dir / collision_name
        counter += 1

    # 4. File Operation (Copy or Rename)
    try:
        if args.copy:
            # shutil.copy2 preserves original metadata during the copy
            shutil.copy2(filepath, new_filepath)
            print(f"  [+] Copied to: {new_filepath}")
        else:
            if filepath.resolve() != new_filepath.resolve():
                filepath.rename(new_filepath)
                print(f"  [+] Renamed to: {new_filepath}")
            else:
                 print(f"  [+] File already named correctly.")
    except Exception as e:
        print(f"  [!] File operation failed: {e}")
        return

    # 5. Inject Tags into the NEW file
    print(f"  [*] Injecting {len(tags)} tags into metadata...")
    inject_metadata_safely(new_filepath, tags)

def main():
    parser = argparse.ArgumentParser(description="AI Image Tagger and Renamer")
    
    # Required positional argument
    parser.add_argument("input_dir", type=str, help="Path to the directory containing images to process.")
    
    # Optional flags
    parser.add_argument("-r", "--recursive", action="store_true", help="Search for images in subdirectories recursively.")
    parser.add_argument("-c", "--copy", action="store_true", help="Copy files to a new name/location instead of renaming the originals.")
    parser.add_argument("-o", "--output_dir", type=str, help="Destination directory if copying. Defaults to the same folder as the image.", default=None)
    parser.add_argument("-m", "--model", type=str, default="gemini-2.5-flash-lite", help="The Gemini model to use (default: gemini-2.5-flash-lite).")
    parser.add_argument("-p", "--prompt", type=str, default="Analyze this image. Provide a description (max 10 words) and 10-20 tags.", help="Custom prompt for the AI.")

    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[!] ERROR: GOOGLE_API_KEY environment variable not set.")
        return

    client = genai.Client(api_key=api_key)
    target_dir = Path(args.input_dir)
    
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"[!] ERROR: Input directory not found: {args.input_dir}")
        return

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    
    # Choose file traversal method based on the recursive flag
    if args.recursive:
        file_iterator = target_dir.rglob("*")
        print(f"[*] Starting recursive scan in: {target_dir}")
    else:
        file_iterator = target_dir.iterdir()
        print(f"[*] Starting scan in: {target_dir}")

    processed_count = 0
    for filepath in file_iterator:
        if filepath.is_file() and filepath.suffix.lower() in valid_extensions:
            process_image(client, filepath, args)
            processed_count += 1
            
    print(f"\n[*] Finished! Processed {processed_count} images.")

if __name__ == "__main__":
    main()