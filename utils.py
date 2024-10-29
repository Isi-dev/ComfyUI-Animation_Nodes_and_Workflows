import torch
import torchaudio
import numpy as np
import cv2



def convert_to_uint8(image):  
    if image.dtype == np.uint8:
        return image
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32)
    if image.max() <= 1.0:
        image = image * 255.0
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)




def resize_image(imageTensor, target_height, target_width):
    
    images = []
    for image in imageTensor:
        print("Resizing Image...")
        image = image.squeeze(0).cpu().numpy()
        image = convert_to_uint8(image)
        image = cv2.resize(image, (int(target_width), int(target_height)), interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()/255
        image = image.permute(0, 2, 3, 1)
        images.append(image)
        
    print ("Images resized!")
    return torch.cat(images, dim=0)




def generate_silent_audio(duration, sample_rate, num_channels=2, batch_size=1):
    """
    Generates a silent audio tensor of the given duration.

    Parameters:
    duration (float): Duration of the silence in seconds (can be fractional).
    sample_rate (int): The audio sample rate (e.g., 44100 Hz).
    num_channels (int): Number of audio channels (1 for mono, 2 for stereo, etc.).
    batch_size (int): Number of audio batches to generate (default is 1).

    Returns:
    torch.Tensor: A 3D tensor filled with zeros, representing silence with shape (batch_size, num_channels, num_samples).
    """
    # Compute the total number of samples based on duration and sample rate
    num_samples = int(round(duration * sample_rate))

    # Return a 3D tensor of zeros (representing silence)
    return torch.zeros((batch_size, 2, num_samples), dtype=torch.float32)



def resample_audio(audio_tensor, target_sample_rate):
    """
    Resamples the audio to the target sample rate if necessary, and ensures a 3D tensor output.

    Parameters:
    audio_tensor (dict): Dictionary containing 'waveform' (audio tensor) and 'sample_rate'.
    target_sample_rate (int): The target sample rate to resample the audio.

    Returns:
    dict: Dictionary containing the resampled 'waveform' (3D tensor) and 'sample_rate'.
    """
    orig_sample_rate = audio_tensor['sample_rate']
    waveform = audio_tensor['waveform']

    # Ensure the waveform has a batch dimension (convert 2D to 3D if necessary)
    if waveform.dim() == 2:  # Shape is (num_channels, num_samples)
        waveform = waveform.unsqueeze(0)  # Add a batch dimension -> (1, num_channels, num_samples)

    # Resample if necessary
    if orig_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=target_sample_rate)
        resampled_waveform = resampler(waveform)
        return {'waveform': resampled_waveform, 'sample_rate': target_sample_rate}
    else:
        return {'waveform': waveform, 'sample_rate': orig_sample_rate}



def get_first_available_sample_rate(audios_list, default_sample_rate=44100):
    """
    Get the sample rate of the first available (non-None) audio from the list.

    Parameters:
    audios_list (list of dict): List of dictionaries containing 'waveform' and 'sample_rate', 
                                where None indicates no audio is present.
    default_sample_rate (int): Default sample rate to return if no valid audio is found.

    Returns:
    int: The sample rate of the first available audio or the default sample rate if none is found.
    """
    for audio_tensor in audios_list:
        if audio_tensor is not None:
            # Extract the sample rate from the dictionary
            print (audio_tensor['sample_rate'])
            return audio_tensor['sample_rate']  # Return the sample rate of the first valid audio
    
    # Return the default sample rate if no valid audio is found
    print("No valid audio sample rate could be determined from the audio list. Using default sample rate.")

    return default_sample_rate




def create_frame_and_audio_lists(frames_set1=None, frames_set2=None, frames_set3=None, frames_set4=None, frames_set5=None, audio_set1=None, audio_set2=None, audio_set3=None, audio_set4=None, audio_set5=None):

    frames_list = [frames_set1, frames_set2, frames_set3, frames_set4, frames_set5]
    audios_list = [audio_set1, audio_set2, audio_set3, audio_set4, audio_set5]
    filtered_frames_list = [frames for frames in frames_list if frames is not None]

    return filtered_frames_list, audios_list




def combine_frames_and_audios(frames_list, audios_list=None, frame_rate=30):
    """
    Combines multiple sets of image frames (as tensors) and audios (as tensors) into a single set.
    Inserts silence for image sets that have no corresponding audio. Resizes and crops images.

    Parameters:
    frames_list (list of torch.Tensor): A list of 4D tensors of image frames (batch_size x channels x height x width).
    audios_list (list of dict, optional): A list of dictionaries, each containing 'waveform' (audio tensor) and 'sample_rate', or None if no audio is present.

    Returns:
    combined_frames (torch.Tensor): A combined 4D tensor of image frames.
    combined_audio (torch.Tensor or None): A combined 3D tensor of audio data, with silence where necessary.
    """
    # Get the target width and height from the first set of frames
    target_height, target_width = frames_list[0].size(1), frames_list[0].size(2)

    # print(f"target_height, target_width =  {frames_list[0].size(1), frames_list[0].size(2)}")

    # Resize and crop each set of frames to match the first set
    resized_frames_list = [resize_image(frames, target_height, target_width) for frames in frames_list]

    # Combine the frames from all videos
    combined_frames = torch.cat(resized_frames_list, dim=0)  # Concatenate along the batch dimension

    # print(video_info)
    # if video_info is not None:
    #     first_key = next(iter(video_info))
    #     first_value = video_info[first_key]

    fps = frame_rate

    # If audios_list is provided, resample and combine audios
    if audios_list is not None:
        # Get the sample rate of the first available audio tensor
        first_audio_sample_rate = get_first_available_sample_rate(audios_list)

        # Prepare a list to store resampled or silent audio tensors
        combined_audios = []
        combined_audios_dict = []

        for i, frames in enumerate(frames_list):

            num_frames = frames.size(0)  # Number of frames in this set
            print(f"There are {num_frames} frames in this video.")
            duration = num_frames / fps  # Duration in seconds based on the frame rate

            # print(f"The duration of this frame is : {duration}")

            audio_dict = audios_list[i]  # The audio dictionary for this set

            if audio_dict is None:
                # Extract the number of channels from the first available audio or default to 1
                num_channels = audios_list[0]['waveform'].size(0) if audios_list[0] is not None else 2
                silent_audio = generate_silent_audio(duration, first_audio_sample_rate, num_channels)
                combined_audios.append(silent_audio)

                print("No audio provided, so silent audio added!")
                # print(f"Silent Audio: {silent_audio.shape}")
            else:
                # Extract the waveform and sample_rate from the audio dictionary
                waveform = audio_dict['waveform']
                sample_rate = audio_dict['sample_rate']
                print(f"Audio provided! Sample rate: {sample_rate}")
                # Resample the audio to match the sample rate of the first audio set
                resampled_audio = resample_audio({'waveform': waveform, 'sample_rate': sample_rate}, first_audio_sample_rate)
                print(f"New Sample rate: {resampled_audio['sample_rate']}")
                # print(f"Audio: {resampled_audio['waveform'].shape}")
                combined_audios.append(resampled_audio['waveform'])  # Append the resampled waveform
                # combined_audios.append(audio_dict['waveform'])

        # Concatenate all audios (real and silent)
        combined_audio = torch.cat(combined_audios, dim=2)  # Concatenate along the sample dimension

        combined_audios_dict = {
            'waveform': combined_audio,
            'sample_rate': first_audio_sample_rate
        }

    else:
        combined_audios_dict = None
        print("No audio provided!")

    return combined_frames, combined_audios_dict


