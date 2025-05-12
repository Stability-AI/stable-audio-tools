import random
import torch
from enum import Enum
from typing import List, Optional, Tuple

class MaskType(Enum):
    RANDOM_SEGMENTS = 0
    FULL_MASK = 1
    CAUSAL_MASK = 2

def random_inpaint_mask(
    sequence: torch.Tensor,
    padding_masks: torch.Tensor,
    max_mask_segments: int = 10,
    mask_type_probabilities: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates random inpainting masks for a batch of latent audio sequences.
    The output inpainting mask has 0 where data should be inpainted, and 1 where data is provided.

    Args:
        sequence: The input sequence tensor of shape (b, c, sequence_length).
        padding_masks: A tensor of shape (b, sequence_length)
                       where 1 indicates real data latents and 0 indicates latents encoding silence padding.
        max_mask_segments: The maximum number of segments for the RANDOM_SEGMENTS mask type.
        mask_type_probabilities: A list of probabilities for choosing each mask type.
                                 The order should correspond to:
                                 [P(RANDOM_SEGMENTS), P(FULL_MASK), P(CAUSAL_MASK)].
                                 If None, defaults to uniform probabilities.

    Returns:
        A tuple containing:
            - masked_sequence: The sequence with masks applied (original sequence where mask is 1,
                               and usually 0 or a placeholder where mask is 0).
            - inpaint_mask: The generated inpainting mask tensor (0 for inpaint, 1 for keep).
    """
    b, _, sequence_length = sequence.size()

    num_mask_types = len(MaskType)
    if mask_type_probabilities is None:
        mask_type_probabilities = [0.1, 0.8, 0.1]
    else:
        if len(mask_type_probabilities) != num_mask_types:
            raise ValueError(
                f"mask_type_probabilities must have {num_mask_types} elements, "
                f"one for each MaskType. Got {len(mask_type_probabilities)}."
            )
        if not torch.isclose(torch.tensor(sum(mask_type_probabilities)), torch.tensor(1.0)):
            raise ValueError(
                f"mask_type_probabilities must sum to 1.0. "
                f"Current sum: {sum(mask_type_probabilities)}"
            )

    output_masks_list = []
    mask_types_to_sample = [mt.value for mt in MaskType]

    for i in range(b):
        padding_mask_single_item = padding_masks[i]
        real_sequence_length = (padding_mask_single_item == 1).sum().item()

        item_mask = torch.ones((1, 1, sequence_length), device=sequence.device, dtype=torch.float32)

        chosen_mask_value = random.choices(mask_types_to_sample, weights=mask_type_probabilities, k=1)[0]
        current_mask_type = MaskType(chosen_mask_value)

        if current_mask_type == MaskType.FULL_MASK:
            item_mask = torch.zeros((1, 1, sequence_length), device=sequence.device, dtype=torch.float32)
        elif real_sequence_length == 0:
            pass # item_mask remains all ones for RANDOM_SEGMENTS/CAUSAL_MASK on empty real data
        else:
            # Logic for RANDOM_SEGMENTS and CAUSAL_MASK when real_sequence_length > 0
            if current_mask_type == MaskType.RANDOM_SEGMENTS:
                num_segments = random.randint(1, max_mask_segments)
                # Max length for a single segment, based on average length
                max_len_per_segment_calc = max(1, real_sequence_length // num_segments)
                
                for _ in range(num_segments):
                    segment_length = random.randint(1, max_len_per_segment_calc)
                    
                    if real_sequence_length - segment_length < 0:
                        continue 
                    mask_start = random.randint(0, real_sequence_length - segment_length)
                    item_mask[:, :, mask_start : mask_start + segment_length] = 0

            elif current_mask_type == MaskType.CAUSAL_MASK:
                # Keep a prefix of real data, inpaint the suffix.
                # The length of the unmasked prefix can be from 0 to real_sequence_length.
                unmasked_prefix_len = random.randint(0, real_sequence_length)
                
                if unmasked_prefix_len < real_sequence_length:
                    item_mask[:, :, unmasked_prefix_len:real_sequence_length] = 0
        
        output_masks_list.append(item_mask)

    final_inpaint_mask = torch.cat(output_masks_list, dim=0).to(sequence.device)
    masked_sequence = sequence * final_inpaint_mask
    return masked_sequence, final_inpaint_mask