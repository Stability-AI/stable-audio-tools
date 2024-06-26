def get_custom_metadata(info, audio):

    # Pass in the relative path of the audio file as the prompt
    return {"prompt": info["relpath"]}
