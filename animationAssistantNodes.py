import cv2
import numpy as np
import torch
from .utils import create_frame_and_audio_lists, combine_frames_and_audios, convert_to_uint8, get_available_styles, generate_prompts
from nodes import MAX_RESOLUTION
import warnings

class JoinVideos():

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {           
                "images_1": ("IMAGE",),            
            },
            "optional": {  
                # "video_info": ("VHS_VIDEOINFO",),          
                "audio_1": ("AUDIO",),          
                "images_2": ("IMAGE",),
                "audio_2": ("AUDIO",),
                "images_3": ("IMAGE",),
                "audio_3": ("AUDIO",),
                "images_4": ("IMAGE",),
                "audio_4": ("AUDIO",),
                "images_5": ("IMAGE",),
                "audio_5": ("AUDIO",),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 1.0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "process"
    CATEGORY = "Animation Assistant Nodes"

    def process(self, images_1, images_2 = None, images_3 = None, images_4 = None, images_5 = None, audio_1 = None, audio_2 = None, audio_3 = None, audio_4 = None, audio_5 = None, fps = 30):

        frames_list, audios_list = create_frame_and_audio_lists(images_1, images_2, images_3, images_4, images_5, audio_1, audio_2, audio_3, audio_4, audio_5)

        images, audio = combine_frames_and_audios(frames_list, audios_list, fps)
        
        print("Combination Complete!")

        return (images, audio)   
    



class Replace_Img_or_Vid_Bg_Assistant():

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "foreground_images": ("IMAGE",),
                "background_images": ("IMAGE",),
                "x_offset": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "y_offset": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "scale_x": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "scale_y": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "bg_scale_x": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "bg_scale_y": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),

            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "process"
    CATEGORY = "image"


    def process(self, foreground_images, background_images, x_offset, y_offset, scale_x, scale_y, bg_scale_x, bg_scale_y):
        if foreground_images is None or background_images is None:
            raise ValueError("Both foreground images and background image are required")

        def ensure_rgba(image):
            if image.ndim == 2:  # Grayscale image
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
            elif image.shape[2] == 3:  # RGB image
                return cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
            elif image.shape[2] == 4:  # RGBA image
                return image
            else:
                raise ValueError("Unsupported image format")

        def scale_image(image, scaleX, scaleY):
            if scaleX != 1.0 or scaleY != 1.0:
                new_size = (int(image.shape[1] * scaleX), int(image.shape[0] * scaleY))
                return cv2.resize(image, new_size, interpolation= cv2.INTER_LINEAR)
            return image

        batch_size = foreground_images.shape[0]

        # Handle cases where only one background image is provided
        if background_images.ndim == 3:
            background_images = np.expand_dims(background_images, axis=0)  # Add batch dimension
        
        num_backgrounds = background_images.shape[0]

        # Raise a warning and handle cases based on the number of background images
        if 1 < num_backgrounds < batch_size:
            warnings.warn(
                "The number of background images is fewer than the number of foreground images. "
                "Repeating the last background image to match the number of foreground images."
            )
            # Repeat the last background image to match the number of foreground images
            background_images = torch.cat(
                [background_images, background_images[-1:].repeat(batch_size - num_backgrounds, 1, 1, 1)], 
                dim=0
            )
        elif num_backgrounds == 1 and batch_size > 1:
            warnings.warn(
                "Only one background image provided. Repeating the background image for all foreground images."
            )
            # If only one background image, replicate it for all foreground images
            background_images = np.repeat(background_images, batch_size, axis=0)
        elif num_backgrounds > batch_size:
            warnings.warn(
                "The number of background images is greater than the number of foreground images. "
                "Reducing the number of background images to match the foreground images."
            )
            # Truncate the background images to match the number of foreground images
            background_images = background_images[:batch_size]

        composite_images = []
        masks = []

        for i in range(batch_size):
            fg_image = foreground_images[i].squeeze(0).cpu().numpy()
            fg_image = ensure_rgba(fg_image)
            fg_image = scale_image(fg_image, scale_x, scale_y)
            fg_height, fg_width = fg_image.shape[:2]

            background_image = background_images[i].numpy()
            background_image = ensure_rgba(background_image)
            background_image = scale_image(background_image, bg_scale_x, bg_scale_y)
            bg_height, bg_width = background_image.shape[:2]

            # Calculate the size of the default background
            default_width = max(bg_width, fg_width) * 3
            default_height = max(bg_height, fg_height) * 3

            # Create a default background (white color)
            default_background = np.ones((default_height, default_width, 4), dtype=np.float32)

            # Calculate offsets to center the input background on the default background
            bg_x_offset = (default_width - bg_width) // 2
            bg_y_offset = (default_height - bg_height) // 2

            # Place the input background on the default background
            default_background[bg_y_offset:bg_y_offset+bg_height, bg_x_offset:bg_x_offset+bg_width] = background_image

            # Calculate offsets to center the foreground on the default background
            fg_x_offset = (default_width - fg_width) // 2 + x_offset
            # fg_y_offset = (default_height - fg_height) // 2 + y_offset
            fg_y_offset = bg_y_offset + bg_height - fg_height + y_offset

            # Ensure the foreground is within the bounds of the default background
            fg_x_offset = max(0, min(fg_x_offset, default_width - fg_width))
            fg_y_offset = max(0, min(fg_y_offset, default_height - fg_height))

            # Composite the foreground onto the default background
            alpha_foreground = fg_image[:, :, 3]
            for c in range(3):
                default_background[fg_y_offset:fg_y_offset+fg_height, fg_x_offset:fg_x_offset+fg_width, c] = (
                    fg_image[:, :, c] * alpha_foreground +
                    default_background[fg_y_offset:fg_y_offset+fg_height, fg_x_offset:fg_x_offset+fg_width, c] * (1 - alpha_foreground)
                )
            
            # Set alpha channel in the composite image
            default_background[fg_y_offset:fg_y_offset+fg_height, fg_x_offset:fg_x_offset+fg_width, 3] = (
                fg_image[:, :, 3] + default_background[fg_y_offset:fg_y_offset+fg_height, fg_x_offset:fg_x_offset+fg_width, 3] * (1 - alpha_foreground)
            )

            # Crop the composite image to the size of the input background
            composite_image = default_background[bg_y_offset:bg_y_offset+bg_height, bg_x_offset:bg_x_offset+bg_width]

            # Convert composite image back to tensor
            output_image_tensor = torch.from_numpy(composite_image).permute(2, 0, 1).unsqueeze(0).float()
            output_image_tensor = output_image_tensor.permute(0, 2, 3, 1)

            composite_images.append(output_image_tensor)

            # Assume a default zero mask for simplicity
            mask = torch.zeros((1, bg_height, bg_width), dtype=torch.float32)
            masks.append(mask)

        # Stack composite images and masks
        composite_images = torch.cat(composite_images, dim=0)
        masks = torch.cat(masks, dim=0)

        return (composite_images, masks)

    


class MakePortraitWalk:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "walk_yoffset": ("INT", {"default": 7, "min": 7, "max": 35, "step": 7}),
                "walk_cycles": ("INT", {"default": 2, "min": 1, "max": 50, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "process"
    CATEGORY = "image"

    def process(self, image, walk_yoffset, walk_cycles):
        if image is None:
            raise ValueError("An image is required")

        # Ensure the background image is properly formatted
        if isinstance(image, torch.Tensor):
            image = image.squeeze(0).cpu().numpy()
        
        if image.ndim not in [2, 3]:
            raise ValueError("Image must be a 2D or 3D numpy array")

        def ensure_rgba(img):
            if img.ndim == 2:  # Grayscale image
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
            elif img.shape[2] == 3:  # RGB image
                return cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            elif img.shape[2] == 4:  # RGBA image
                return img
            else:
                raise ValueError("Unsupported image format")

        image = ensure_rgba(image)
        image_height, image_width = image.shape[:2]

        imageDiffY = int(walk_yoffset/7)

        shifted_image = np.zeros_like(image)
        shifted_image[walk_yoffset:, :] = image[:-walk_yoffset, :]

        half_walk_cycle_images = []
        all_images = []
        masks = []

        #Standing Images
        for i in range(5):
            half_walk_cycle_images.append(shifted_image)

        # Going down images
        going_down_images = []
        for i in range(7):
            x = imageDiffY * (i + 1)
            goDown_image = np.zeros_like(image)
            goDown_image[x:, :] = image[:-x, :]
            going_down_images.append(goDown_image)

        # Going up images (reverse of going down images)
        going_up_images = list(reversed(going_down_images))
        half_walk_cycle_images.extend(going_up_images)

        # Standing Image
        half_walk_cycle_images.append(image)

        # Add the going down images to the half walk cycle
        half_walk_cycle_images.extend(going_down_images)

        full_walk_cycle_images = half_walk_cycle_images * 2

        total_walk_cycles = full_walk_cycle_images * walk_cycles

        for i, frame in enumerate(total_walk_cycles):

            # Convert image back to tensor
            output_image_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
            output_image_tensor = output_image_tensor.permute(0, 2, 3, 1)

            all_images.append(output_image_tensor)

            # Assume a default zero mask for simplicity
            mask = torch.zeros((1, image_height, image_width), dtype=torch.float32)
            masks.append(mask)

        # Stack composite images and masks
        all_images = torch.cat(all_images, dim=0)
        masks = torch.cat(masks, dim=0)

        return (all_images, masks)



class MoveInOrOut:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foreground_images": ("IMAGE",),  # Batch of images
                "background_image": ("IMAGE",),   # Single background image
                "scaling_factor": ("FLOAT", {"default": 2.0, "min": 1.5, "max": 20.0}),
                "move_in": ("BOOLEAN", { "default": False }),
                "fg_x_offset": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "fg_y_offset": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "process"
    CATEGORY = "image"

    def process(self, foreground_images, background_image, scaling_factor, move_in, fg_x_offset, fg_y_offset):
        if foreground_images is None or background_image is None:
            raise ValueError("Both foreground images and background image are required")

        if scaling_factor < 1.5:
            raise ValueError("Scaling factor must be greater than or equal to 1.5")

        # Ensure the background image is properly formatted
        if isinstance(background_image, torch.Tensor):
            background_image = background_image.squeeze(0).cpu().numpy()
        
        if background_image.ndim not in [2, 3]:
            raise ValueError("Background image must be a 2D or 3D numpy array")

        def ensure_rgba(image):
            if image.ndim == 2:  # Grayscale image
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
            elif image.shape[2] == 3:  # RGB image
                return cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
            elif image.shape[2] == 4:  # RGBA image
                return image
            else:
                raise ValueError("Unsupported image format")

        background_image = ensure_rgba(background_image)
        bg_height, bg_width = background_image.shape[:2]
        fg_count = len(foreground_images)

        # Scale the background image to its maximum
        scaled_bg_height = int(bg_height * scaling_factor)
        scaled_bg_width = int(bg_width * scaling_factor)
        scaled_bg_height_reduction_factor = (scaled_bg_height - bg_height)/(fg_count - 1)
        scaled_bg_width_reduction_factor = (scaled_bg_width - bg_width)/(fg_count - 1)

        composite_images = []
        masks = []

        for i, fg_image in enumerate(foreground_images):
            if isinstance(fg_image, torch.Tensor):
                fg_image = fg_image.squeeze(0).cpu().numpy()

            if fg_image.ndim not in [2, 3]:
                raise ValueError("Foreground images must be 2D or 3D numpy arrays")

            fg_image = ensure_rgba(fg_image)
            fg_height, fg_width = fg_image.shape[:2]


            if move_in:
                current_scaled_bg_height = int(bg_height + scaled_bg_height_reduction_factor*i)
                current_scaled_bg_width = int(bg_width + scaled_bg_width_reduction_factor*i)
            else:
                current_scaled_bg_height = int(scaled_bg_height - scaled_bg_height_reduction_factor*i)
                current_scaled_bg_width = int(scaled_bg_width - scaled_bg_width_reduction_factor*i)



            scaled_background = cv2.resize(background_image, (current_scaled_bg_width, current_scaled_bg_height), interpolation=cv2.INTER_LINEAR)

            # Calculate cropping dimensions for each frame to create the illusion of motion
            crop_x1 = int((current_scaled_bg_width - bg_width)/2)
            crop_y1 = int((current_scaled_bg_height - bg_height)/2)
            cropped_background = scaled_background[crop_y1:crop_y1+bg_height, crop_x1:crop_x1+bg_width]

            # Default offsets
            x_offset = (bg_width - fg_width) // 2
            y_offset = bg_height - fg_height

            if fg_x_offset < 0:
                x_offset = max(0, x_offset + fg_x_offset)
            if fg_x_offset > 0:
                x_offset = min(bg_width - fg_width, x_offset + fg_x_offset)
            if fg_y_offset < 0:
                y_offset = max(0, y_offset + fg_y_offset)

            # Ensure the dimensions are within bounds
            y1, y2 = y_offset, y_offset + fg_height
            x1, x2 = x_offset, x_offset + fg_width

            # Composite the images
            composite_image = cropped_background.copy()
            alpha_foreground = fg_image[:, :, 3]
            alpha_background = cropped_background[y1:y2, x1:x2, 3]
            alpha_composite = alpha_foreground + alpha_background * (1 - alpha_foreground)

            for c in range(3):
                composite_image[y1:y2, x1:x2, c] = (
                    fg_image[:, :, c] * alpha_foreground +
                    cropped_background[y1:y2, x1:x2, c] * (1 - alpha_foreground)
                )
            
            # Set alpha channel in the composite image
            composite_image[y1:y2, x1:x2, 3] = alpha_composite

            # Convert composite image back to tensor
            output_image_tensor = torch.from_numpy(composite_image).permute(2, 0, 1).unsqueeze(0).float()
            output_image_tensor = output_image_tensor.permute(0, 2, 3, 1)

            composite_images.append(output_image_tensor)

            # Assume a default zero mask for simplicity
            mask = torch.zeros((1, bg_height, bg_width), dtype=torch.float32)
            masks.append(mask)

        # Stack composite images and masks
        composite_images = torch.cat(composite_images, dim=0)
        masks = torch.cat(masks, dim=0)

        return (composite_images, masks)




class MoveLeftOrRight:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foreground_images": ("IMAGE",),  # Batch of images
                "background_image": ("IMAGE",),   # Single background image
                "move_right": ("BOOLEAN", { "default": False }),
                "speed": ("INT", {"default": 1, "min": 1, "max": 50, "step": 1}),
                "fg_x_offset": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "fg_y_offset": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "view_width": ("INT", {"default": 768, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "process"
    CATEGORY = "image"

    def process(self, foreground_images, background_image, move_right, speed, fg_x_offset, fg_y_offset, view_width):
        if foreground_images is None or background_image is None:
            raise ValueError("Both foreground images and background image are required")

        # Ensure the background image is properly formatted
        if isinstance(background_image, torch.Tensor):
            background_image = background_image.squeeze(0).cpu().numpy()
        
        if background_image.ndim not in [2, 3]:
            raise ValueError("Background image must be a 2D or 3D numpy array")

        def ensure_rgba(image):
            if image.ndim == 2:  # Grayscale image
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
            elif image.shape[2] == 3:  # RGB image
                return cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
            elif image.shape[2] == 4:  # RGBA image
                return image
            else:
                raise ValueError("Unsupported image format")

        background_image = ensure_rgba(background_image)
        bg_height, bg_width = background_image.shape[:2]


        distance = bg_width - view_width

        bg_x_start = distance
        if move_right:
            bg_x_start = 0

        composite_images = []
        masks = []


        if foreground_images.shape[0] == 1:
            print("Only one foreground provided. Increasing to match speed...")
            increase_single_fg_number = distance // speed
            foreground_images = np.repeat(foreground_images, increase_single_fg_number, axis=0)


        for i, fg_image in enumerate(foreground_images):
            if isinstance(fg_image, torch.Tensor):
                fg_image = fg_image.squeeze(0).cpu().numpy()

            if fg_image.ndim not in [2, 3]:
                raise ValueError("Foreground images must be 2D or 3D numpy arrays")

            fg_image = ensure_rgba(fg_image)
            fg_height, fg_width = fg_image.shape[:2]


            if move_right:

                bg_x_start = int(min(distance - 1, bg_x_start+speed))
                
            else:
                bg_x_start = int(max(0, bg_x_start-speed))

            # Calculate cropping dimensions for each frame to create the illusion of motion
            crop_x1 = bg_x_start
            crop_y1 = 0
            cropped_background = background_image[crop_y1:crop_y1+bg_height, crop_x1:crop_x1+view_width]

            # Default offsets
            x_offset = (view_width - fg_width) // 2
            y_offset = bg_height - fg_height

            if fg_x_offset < 0:
                x_offset = max(0, x_offset + fg_x_offset)
            if fg_x_offset > 0:
                x_offset = min(view_width - fg_width, x_offset + fg_x_offset)
            if fg_y_offset < 0:
                y_offset = max(0, y_offset + fg_y_offset)

            # Ensure the dimensions are within bounds
            y1, y2 = y_offset, y_offset + fg_height
            x1, x2 = x_offset, x_offset + fg_width

            # Composite the images
            composite_image = cropped_background.copy()
            alpha_foreground = fg_image[:, :, 3]
            alpha_background = cropped_background[y1:y2, x1:x2, 3]
            alpha_composite = alpha_foreground + alpha_background * (1 - alpha_foreground)

            for c in range(3):
                composite_image[y1:y2, x1:x2, c] = (
                    fg_image[:, :, c] * alpha_foreground +
                    cropped_background[y1:y2, x1:x2, c] * (1 - alpha_foreground)
                )
            
            # Set alpha channel in the composite image
            composite_image[y1:y2, x1:x2, 3] = alpha_composite

            # Convert composite image back to tensor
            output_image_tensor = torch.from_numpy(composite_image).permute(2, 0, 1).unsqueeze(0).float()
            output_image_tensor = output_image_tensor.permute(0, 2, 3, 1)

            composite_images.append(output_image_tensor)

            # Assume a default zero mask for simplicity
            mask = torch.zeros((1, bg_height, bg_width), dtype=torch.float32)
            masks.append(mask)

        # Stack composite images and masks
        composite_images = torch.cat(composite_images, dim=0)
        masks = torch.cat(masks, dim=0)

        return (composite_images, masks)



class MoveUpOrDown:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foreground_images": ("IMAGE",),  # Batch of images
                "background_image": ("IMAGE",),   # Single background image
                "move_down": ("BOOLEAN", { "default": False }),
                "speed": ("INT", {"default": 1, "min": 1, "max": 50, "step": 1}),
                "fg_x_offset": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "fg_y_offset": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "view_height": ("INT", {"default": 768, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "process"
    CATEGORY = "image"

    def process(self, foreground_images, background_image, move_down, speed, fg_x_offset, fg_y_offset, view_height):
        if foreground_images is None or background_image is None:
            raise ValueError("Both foreground images and background image are required")

        # Ensure the background image is properly formatted
        if isinstance(background_image, torch.Tensor):
            background_image = background_image.squeeze(0).cpu().numpy()
        
        if background_image.ndim not in [2, 3]:
            raise ValueError("Background image must be a 2D or 3D numpy array")

        def ensure_rgba(image):
            if image.ndim == 2:  # Grayscale image
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
            elif image.shape[2] == 3:  # RGB image
                return cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
            elif image.shape[2] == 4:  # RGBA image
                return image
            else:
                raise ValueError("Unsupported image format")

        background_image = ensure_rgba(background_image)
        bg_height, bg_width = background_image.shape[:2]


        distance = bg_height - view_height

        bg_y_start = distance
        if move_down:
            bg_y_start = 0

        composite_images = []
        masks = []


        if foreground_images.shape[0] == 1:
            print("Only one foreground provided. Increasing to match speed...")
            increase_single_fg_number = distance // speed
            foreground_images = np.repeat(foreground_images, increase_single_fg_number, axis=0)
        


        for i, fg_image in enumerate(foreground_images):
            if isinstance(fg_image, torch.Tensor):
                fg_image = fg_image.squeeze(0).cpu().numpy()

            if fg_image.ndim not in [2, 3]:
                raise ValueError("Foreground images must be 2D or 3D numpy arrays")

            fg_image = ensure_rgba(fg_image)
            fg_height, fg_width = fg_image.shape[:2]


            if move_down:

                bg_y_start = int(min(distance - 1, bg_y_start+speed))
                
            else:
                bg_y_start = int(max(0, bg_y_start-speed))

            # Calculate cropping dimensions for each frame to create the illusion of motion
            crop_x1 = 0
            crop_y1 = bg_y_start
            cropped_background = background_image[crop_y1:crop_y1+view_height, crop_x1:crop_x1+bg_width]

            # Default offsets
            x_offset = (bg_width - fg_width) // 2
            y_offset = view_height - fg_height
            if move_down:
                y_offset = 0

            if fg_x_offset < 0:
                x_offset = max(0, x_offset + fg_x_offset)
            if fg_x_offset > 0:
                x_offset = min(bg_width - fg_width, x_offset + fg_x_offset)
            if fg_y_offset < 0:
                y_offset = max(0, y_offset + fg_y_offset)
            if fg_y_offset > 0:
                y_offset = min(view_height - fg_height, y_offset + fg_y_offset)

            # Ensure the dimensions are within bounds
            y1, y2 = y_offset, y_offset + fg_height
            x1, x2 = x_offset, x_offset + fg_width

            # Composite the images
            composite_image = cropped_background.copy()
            alpha_foreground = fg_image[:, :, 3]
            alpha_background = cropped_background[y1:y2, x1:x2, 3]
            alpha_composite = alpha_foreground + alpha_background * (1 - alpha_foreground)

            for c in range(3):
                composite_image[y1:y2, x1:x2, c] = (
                    fg_image[:, :, c] * alpha_foreground +
                    cropped_background[y1:y2, x1:x2, c] * (1 - alpha_foreground)
                )
            
            # Set alpha channel in the composite image
            composite_image[y1:y2, x1:x2, 3] = alpha_composite

            # Convert composite image back to tensor
            output_image_tensor = torch.from_numpy(composite_image).permute(2, 0, 1).unsqueeze(0).float()
            output_image_tensor = output_image_tensor.permute(0, 2, 3, 1)

            composite_images.append(output_image_tensor)

            # Assume a default zero mask for simplicity
            mask = torch.zeros((1, bg_height, bg_width), dtype=torch.float32)
            masks.append(mask)

        # Stack composite images and masks
        composite_images = torch.cat(composite_images, dim=0)
        masks = torch.cat(masks, dim=0)

        return (composite_images, masks)
    


class MakeDrivingVideoForLivePortrait:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),  # Batch of images
                "handle_rot": ("BOOLEAN", { "default": False }),
                "set_face_box": ("BOOLEAN", { "default": False }),
                "box_size": ("INT", {"default": 256, "min": 128, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("video", "video2")
    FUNCTION = "process"
    CATEGORY = "image"

    def process(self, video, handle_rot, set_face_box, box_size):

        mp = import_mediapipe()

        if mp is None:
            print("MediaPipe could not be imported or installed. Please check your environment and try installing it.")
            return

        if video is None:
            raise ValueError("Input video is required")
        
        mp_face_mesh = mp.solutions.face_mesh

        width = video.shape[2]
        height = video.shape[1]

        shortest_side = min(width, height)

        if shortest_side != 1024:
            scale_factor = 1024 / shortest_side
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            width, height = new_width, new_height  # Adjust the video frame size to the new scaled size
        else:
            new_width = width
            new_height = height

        reference_vector = None

        frames = []

        with mp_face_mesh.FaceMesh(static_image_mode=False,
                               max_num_faces=1,
                               min_detection_confidence=0.5) as face_mesh:
            
            for i, frame in enumerate(video):

                print(f"Detecting face & Converting frame {i} ...")

                if isinstance(frame, torch.Tensor):
                    frame = frame.squeeze(0).cpu().numpy()
                    frame = convert_to_uint8(frame)

                # Resize the frame if scaling is needed
                if shortest_side != 1024:
                    frame = cv2.resize(frame, (new_width, new_height))

                h, w, _ = frame.shape

                # Convert the BGR image to RGB
                # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame)

                if results.multi_face_landmarks:
                    # Get the landmarks for the first detected face
                    landmarks = results.multi_face_landmarks[0]

                    # Extract landmark positions for eyes and mouth
                    eye_left_index = [33, 133, 160]  # Left eye landmarks
                    eye_right_index = [362, 263, 387]  # Right eye landmarks
                    mouth_index = [13]  # Mouth center index
                    forehead_landmarks = [10, 109, 152, 234, 249, 263, 359, 374]  # Adjust indices as needed
                    total_y = 0
                    for idx in forehead_landmarks:
                        total_y += landmarks.landmark[idx].y
                    avg_y = total_y / len(forehead_landmarks)


                    eye_left = np.mean([[landmarks.landmark[i].x * w,
                                        landmarks.landmark[i].y * h] for i in eye_left_index], axis=0)

                    eye_right = np.mean([[landmarks.landmark[i].x * w,
                                        landmarks.landmark[i].y * h] for i in eye_right_index], axis=0)

                    mouth = np.mean([[landmarks.landmark[i].x * w,
                                    landmarks.landmark[i].y * h] for i in mouth_index], axis=0)

                    # Calculate the midpoint between the eyes
                    eye_center = (eye_left + eye_right) / 2

                    # Create the vector from the eye midpoint to the mouth
                    current_vector = mouth - eye_center

                    if reference_vector is None:
                        # Save the vector from the first frame as the reference
                        reference_vector = current_vector
                        diff_factor = abs((mouth[1] - avg_y)/2)
                        if set_face_box:
                            diff_factor = box_size
                       


                    # Calculate the angle between the current vector and the reference vector
                    # angle_reference = np.arctan2(reference_vector[1], reference_vector[0])
                    angle_reference = np.arctan2(reference_vector[1], 0)
                    angle_current = np.arctan2(current_vector[1], current_vector[0])
                    angle_difference = (angle_current - angle_reference) * (180 / np.pi)

                    # Get the rotation matrix to align the current vector with the reference vector
                    M_rot = cv2.getRotationMatrix2D((int(eye_center[0]), int(eye_center[1])), angle_difference, 1.0)

                    # Apply the rotation to the frame
                    rotated_frame = cv2.warpAffine(frame, M_rot, (w, h))

                    # diff_factor = abs((mouth[1] - avg_y)/1.6)
                    # if set_face_box:
                    #     diff_factor = half_size
                    # Crop around the head to keep the head and shoulder centered in the frame
                    
                    x1 = max(int(eye_center[0]) - int(diff_factor), 0)
                    x2 = min(int(eye_center[0]) + int(diff_factor), w)
                    y1 = max(int(eye_center[1]) - int(diff_factor), 0)
                    y2 = min(int(eye_center[1]) + int(diff_factor), h)


                    if handle_rot:
                        cropped_frame = rotated_frame[y1:y2, x1:x2]
                    else:
                        cropped_frame = frame[y1:y2, x1:x2]

                    if cropped_frame.shape[0] > 0 and cropped_frame.shape[1] > 0:
                        resized_frame = cv2.resize(cropped_frame, (512, 512))
                    else:
                        resized_frame = np.zeros((512, 512, 3), dtype=np.uint8)

                    # Write the processed frame to the output video
                    image = torch.from_numpy(resized_frame).permute(2, 0, 1).unsqueeze(0).float()/255
                    image = image.permute(0, 2, 3, 1)
                    frames.append(image)

                else:
                    print("No face detected! Adding blank frame...")
                    blank_frame = np.zeros((512, 512, 3), dtype=np.uint8)
                    image = torch.from_numpy(blank_frame).permute(2, 0, 1).unsqueeze(0).float()/255
                    image = image.permute(0, 2, 3, 1)
                    frames.append(image)
                    

        return torch.cat(frames, dim=0), torch.cat(frames, dim=0)

    


class CLIPTextEncodeStyles:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}), 
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "style": (get_available_styles(),),
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."})
            }
        }
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("+VE", "-VE")
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images with selected style."

    def encode(self, clip, positive_prompt, negative_prompt, style):
        prompts = generate_prompts(style, positive_prompt, negative_prompt)
        tokens = clip.tokenize(prompts["positive_prompt"])
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        tokensN = clip.tokenize(prompts["negative_prompt"])
        outputN = clip.encode_from_tokens(tokensN, return_pooled=True, return_dict=True)
        condN = outputN.pop("cond")
        return ([[cond, output]], [[condN, outputN]],)
    




def import_mediapipe():
    import subprocess
    import sys

    try:
        import mediapipe as mp
        print("MediaPipe is successfully imported.")
        return mp
    except ImportError:
        print("MediaPipe is not installed.")
        
        # Check if pip is available
        try:
            import pip
            print("Attempting to install MediaPipe...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapipe"])
            
            # Try importing again
            import mediapipe as mp
            print("MediaPipe has been installed and imported successfully.")
            return mp
        except ImportError:
            print("Pip is not available in this environment.")
        except Exception as e:
            print(f"Failed to install MediaPipe: {e}")
    
    return None




NODE_CLASS_MAPPINGS = {
    "JoinVideos" : JoinVideos,
    "Replace_Img_or_Vid_Bg_Assistant" : Replace_Img_or_Vid_Bg_Assistant,
    "MakePortraitWalk" : MakePortraitWalk,
    "MoveInOrOut" : MoveInOrOut,
    "MoveLeftOrRight" : MoveLeftOrRight,
    "MoveUpOrDown" : MoveUpOrDown,
    "MakeDrivingVideoForLivePortrait" : MakeDrivingVideoForLivePortrait,
    "CLIPTextEncodeStyles" : CLIPTextEncodeStyles,
    
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "JoinVideos" :"Join Videos",
    "Replace_Img_or_Vid_Bg_Assistant" : "Replace Img_or_Vid_Bg",
    "MakePortraitWalk" : "Make Portrait Walk",
    "MoveInOrOut" : "Move In_Or_Out",
    "MoveLeftOrRight" : "Move Left_Or_Right",
    "MoveUpOrDown" : "Move Up_Or_Down",
    "MakeDrivingVideoForLivePortrait" : "Video for LivePortrait",
    "CLIPTextEncodeStyles" : "Prompt Style Encoder",
    
}
