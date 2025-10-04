import os
import re


def get_custom_metadata(info, audio):
    # Get filename without extension
    file_name = os.path.basename(info["relpath"])
    file_name_without_extension = os.path.splitext(file_name)[0]

    # Replace non-alphanumeric characters with spaces, and remove leading/trailing spaces
    #cleaned_file_name = re.sub('[^0-9a-zA-Z]+', ' ', file_name_without_extension).strip()
    #cleaned_file_name = re.match('', cleaned_file_name).groups()[0]

    # Get parent directory name (without the full path)
    dir_name = os.path.dirname(info["relpath"])
    prompt = os.path.split(dir_name)[1]

    # Use the filename instead of parent directory if the filename has relevant info
    if 'BPM' in prompt:
        prompt = file_name_without_extension

    # Translate X beats of Y notes per bar from XbYn to "normal" time signature notation
    # 4b4n = 4/4
    # 3b16n = 3/16
    # 69b420n = 69/420
    prompt = re.sub(r'(\d+)b(\d+)n', r'\1/\2', prompt)

    # Instrument123 => Instrument
    # Acid1 => Acid
    prompt = re.sub(r'^(\w+)\d+', r'\1', prompt, count=1)

    # Acid DistSplinterFat 120BPM 4/4 4bars
    # Acid Distorted 120BPM 4/4 4bars
    prompt = re.sub(r'Dist\w+ ', r' Distorted ', prompt, count=1)

    # Acid Distorted 120BPM 4/4 4bars
    # Acid Distorted 120 BPM 4/4 4bars
    prompt = re.sub(r'(\d+)BPM', r'\1 BPM', prompt, count=1)

    # Am = A minor
    # G#m = G# minor
    prompt = re.sub(r'( [ABCDEFG][#♭♮♯]?)m ', r'\1 minor ', prompt)
    # AMajor = A Major
    # F#Phrygian = F# Phrygian
    prompt = re.sub(r'(^\w+ [ABCDEFG][#♭♮♯]?)([a-zA-Z]+)', r'\1 \2', prompt, count=1)
    # TODO: obviate this hack
    prompt = re.sub(r' D istorted ', r' Distorted ', prompt, count=1)

    # 4bars = 4 bars
    prompt = re.sub(r'(\d+)bars', r'\1 bars', prompt)

    # Remove (1), (2), etc.
    prompt = re.sub(r'\(\d+\)$', r'', prompt)

    # Sanity check
    print(f'{info["relpath"]} => {prompt}')

    return {"prompt": prompt}
