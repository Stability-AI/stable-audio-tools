import os
import re


def get_custom_metadata(info, audio):
    # Get filename without extension
    file_name = os.path.basename(info["relpath"])
    file_name_without_extension = os.path.splitext(file_name)[0]

    # Replace non-alphanumeric characters with spaces, and remove leading/trailing spaces
    cleaned_file_name = re.sub('[^0-9a-zA-Z]+', ' ', file_name_without_extension).strip()
    #cleaned_file_name = re.match('', cleaned_file_name).groups()[0]

    # Sanity check
    print(f'{info["relpath"]} => {cleaned_file_name}')

    return {"prompt": cleaned_file_name}
