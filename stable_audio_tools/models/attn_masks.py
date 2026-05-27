def generate_sliding_window_mask(sliding_window):
    def sliding_window_mask(b, h, q_idx, kv_idx):
        offset = kv_idx - q_idx
        mask = (offset >= -sliding_window[0]) & (offset <= sliding_window[1])
        return mask
    return sliding_window_mask

def generate_chunked_sliding_window_mask(chunk_size, chunk_sliding_window):
    sequence_start = 0
    def chunked_sliding_window_mask(b, h, q_idx, kv_idx):

        q_chunk_id = (q_idx - sequence_start) // chunk_size
        kv_chunk_id = (kv_idx - sequence_start) // chunk_size

        chunk_offset = kv_chunk_id - q_chunk_id
        mask = (chunk_offset >= -chunk_sliding_window[0]) & (chunk_offset <= chunk_sliding_window[1])

        return mask

    return chunked_sliding_window_mask
