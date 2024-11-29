import torch
import torchaudio
import numpy as np
import cv2
import json
import os



def convert_to_uint8(image):  
    if image.dtype == np.uint8:
        return image
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32)
    if image.max() <= 1.0:
        image = image * 255.0
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)

def convert_to_unit8_v2(image):
    image = (image - image.min()) / (image.max() - image.min())*255

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



def get_script_dir():
    """Get the directory where the script is located."""
    return os.path.dirname(os.path.abspath(__file__))

def load_styles():
    """Load the styles.json file from the script's directory."""
    script_dir = get_script_dir()
    styles_path = os.path.join(script_dir, "styles.json")

    try:
        with open(styles_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The styles.json file was not found in {styles_path}.")
        # Optional: return default styles as a fallback
        return {
            "none": {
                "positive": "",
                "negative": ""
            },
            "anime1": {
                "positive": "Japanese comic book anime style, samurai x,  exaggerated features, sharp dark lines, anime, anime style, colorful, vibrant, detailed, intricate, large expressive eyes",
                "negative": "text, logo, signature, watermark, soft edges, low quality, blurry, realistic style, distorted faces, bad anatomy, poor proportions"
            },
            "anime2": {
                "positive": "a beautiful anime-style illustration, expressive eyes, exaggerated features, vibrant colors, detailed, expressive characters, studio ghibli influence",
                "negative": "low quality, blurry, realistic style, distorted faces, bad anatomy, poor proportions"
            },
            "beauArt": {
                "positive": "hard edges, black and white book illustration, bold lines, black lines, white background",
                "negative": "color, poor contrast, realistic, soft edges"
            },
            "black Line art":{
                "positive": "Black Line art, black and white, sharp lines, clear, minimalist, graphic, precision, ink, monochrome",
                "negative": "photograph, stock photo, realistic, deformed, glitch, color, vague, blurry, noisy, low contrast, photorealistic, realism, impressionism, expressionism, oil, acrylic, watercolor, pastel, textured, gradient, shaded"
            },
            "comic1": {
                "positive": "vivid comic book illustration, bold lines, halftone shading, dramatic action",
                "negative": "washed-out colors, lack of contrast, soft edges, overly detailed backgrounds"
            },
            "comic2": {
                "positive": "comic book style, dynamic pose, exaggerated features, bold lines, vibrant colors, ink wash, retro style, comic book panel",
                "negative": "blurry, low quality, poorly drawn, deformed, disfigured, extra limbs, missing limbs, extra fingers, missing fingers, unrealistic anatomy, poorly proportioned, monochrome, grayscale, black and white"
            },
            "disney": {
                "positive": "classic Disney-style art, colorful, detailed characters, expressive emotions, smooth animations, magical and whimsical setting",
                "negative": "dull colors, lack of detail, distorted features, dark themes"
            },
            "manga": {
                "positive": "black and white manga illustration, bold lines, dynamic compositions, detailed linework, expressive characters, action-packed panels",
                "negative": "color, poor contrast, lack of detail, blurry, distorted anatomy, flat expressions"
            },
            "ps1": {
                "positive": "low-poly 3D graphics in PS1 style, retro gaming aesthetics, simple geometric shapes, nostalgic vibe",
                "negative": "modern high-resolution graphics, realistic textures, excessive detail, smooth rendering"
            },
            "pixar": {
                "positive": "a 3D animation in Pixar style, highly detailed, vibrant, whimsical, expressive characters, cinematic lighting, storybook atmosphere",
                "negative": "grainy textures, flat colors, dull, uninspired composition, poor rendering"
            },
            "toy": {
                "positive": "toy-like miniature characters, bright colors, soft plastic textures, whimsical design, playful atmosphere, detailed craftsmanship",
                "negative": "realistic details, dull colors, rough textures, lack of charm"
            }, 
            "watercolor": {
                "positive": "soft and vibrant watercolor painting, gentle gradients, flowing textures, light and airy feel, pastel tones",
                "negative": "harsh lines, digital appearance, over-saturated colors, poor blending"
            }
    
        }


def get_available_styles():
    styles = load_styles()
    return list(styles.keys())


def generate_prompts(art_style, user_positive_prompt="sharp image", user_negative_prompt="blurry"):
    styles = load_styles()
    
    if art_style not in styles:
        return {
            "positive_prompt": f"Sorry, no predefined prompts for the '{art_style}' style. Please specify another style.",
            "negative_prompt": "None"
        }

    style = styles[art_style]
    combined_positive = f"{style['positive']}, {user_positive_prompt}".strip(", ")
    combined_negative = f"{style['negative']}, {user_negative_prompt}".strip(", ")

    return {
        "positive_prompt": combined_positive,
        "negative_prompt": combined_negative
    }


