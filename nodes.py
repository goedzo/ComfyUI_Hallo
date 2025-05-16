import os
import cv2
import yaml
import torch
import random
import torchaudio
import folder_paths
import numpy as np
from PIL import Image
import sys  # <--- ADD THIS
import subprocess # <--- ADD THIS


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

# Our any instance wants to be a wildcard string
any = AnyType("*")


def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def cv_frame_generator(video):
    try:
        video_cap = cv2.VideoCapture(video)
        if not video_cap.isOpened():
            raise ValueError(f"{video} could not be loaded with cv.")
        # set video_cap to look at start_index frame
        total_frame_count = 0
        total_frames_evaluated = -1
        frames_added = 0
        base_frame_time = 1/video_cap.get(cv2.CAP_PROP_FPS)
        width = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        prev_frame = None
        target_frame_time = base_frame_time
        yield (width, height, target_frame_time)
        time_offset=target_frame_time - base_frame_time
        while video_cap.isOpened():
            if time_offset < target_frame_time:
                is_returned = video_cap.grab()
                # if didn't return frame, video has ended
                if not is_returned:
                    break
                time_offset += base_frame_time
            if time_offset < target_frame_time:
                continue
            time_offset -= target_frame_time
            # if not at start_index, skip doing anything with frame
            total_frame_count += 1
            total_frames_evaluated += 1

            # opencv loads images in BGR format (yuck), so need to convert to RGB for ComfyUI use
            # follow up: can videos ever have an alpha channel?
            # To my testing: No. opencv has no support for alpha
            unused, frame = video_cap.retrieve()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # convert frame to comfyui's expected format
            # TODO: frame contains no exif information. Check if opencv2 has already applied
            frame = np.array(frame, dtype=np.float32) / 255.0
            if prev_frame is not None:
                inp  = yield prev_frame
                if inp is not None:
                    #ensure the finally block is called
                    return
            prev_frame = frame
            frames_added += 1

        if prev_frame is not None:
            yield prev_frame
    finally:
        video_cap.release()

    
class HalloNode:
    @classmethod
    def INPUT_TYPES(s):
        audio_extensions = ["wav", "mp3", "flac"]
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1] in audio_extensions):
                    files.append(f)
        return {"required": {
                    "source_image": ("IMAGE", ),
                    "driving_audio": ("AUDIO", ),
                    "pose_weight" :("FLOAT",{"default": 1.0}),
                    "face_weight":("FLOAT",{"default": 1.0}),
                    "lip_weight":("FLOAT",{"default": 1.0}),
                    "face_expand_ratio":("FLOAT",{"default": 1.2}),
                     },}

    CATEGORY = "HalloNode"

    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", )
    RETURN_NAMES = ("images", "count", "frame_rate", )
    FUNCTION = "inference"


    def inference(self, source_image, driving_audio, pose_weight, face_weight, lip_weight, face_expand_ratio):
        ckpt_dir = os.path.join(folder_paths.models_dir, "hallo")
        cur_dir = get_ext_dir()
        output_dir = folder_paths.get_temp_directory()
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(ckpt_dir):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="fudan-generative-ai/hallo", local_dir=ckpt_dir, local_dir_use_symlinks=False)

        infer_py = os.path.join(cur_dir, "scripts/inference.py")
        default_yaml_path = os.path.join(cur_dir, "configs/inference/default.yaml")
        with open(default_yaml_path, 'r', encoding="utf-8") as f:
            yaml_data = yaml.load(f.read(),Loader=yaml.SafeLoader)
        yaml_data['save_path'] = output_dir
        yaml_data['audio_ckpt_dir'] = os.path.join(ckpt_dir, "hallo")
        yaml_data['base_model_path'] = os.path.join(ckpt_dir, "stable-diffusion-v1-5")
        yaml_data['motion_module_path'] = os.path.join(ckpt_dir, "motion_module/mm_sd_v15_v2.ckpt")
        yaml_data['face_analysis']['model_path'] = os.path.join(ckpt_dir, "face_analysis")
        yaml_data['wav2vec']['model_path'] = os.path.join(ckpt_dir, "wav2vec/wav2vec2-base-960h")
        yaml_data['audio_separator']['model_path'] = os.path.join(ckpt_dir, "audio_separator/Kim_Vocal_2.onnx")
        yaml_data['vae']['model_path'] = os.path.join(ckpt_dir, "sd-vae-ft-mse")
        yaml_data["face_landmarker"]['model_path'] = os.path.join(ckpt_dir, "face_analysis/models/face_landmarker_v2_with_blendshapes.task")

        tmp_yaml_path = os.path.join(cur_dir, 'tmp.yaml')
        with open(tmp_yaml_path, 'w', encoding="utf-8") as f:
            yaml.dump(data=yaml_data, stream=f, Dumper=yaml.Dumper)

        output_name = ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))
        output_video_path = os.path.join(output_dir, f"hallo_{output_name}.mp4")

        # get src image
        for (_, img) in enumerate(source_image):
            img = 255. * img.cpu().numpy()
            img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            src_img_path = os.path.join(output_dir, f"hallo_{output_name}_src_img.png")
            img.save(src_img_path)
            print(f'saved src image to {src_img_path}')
            break

        # # get src audio
        # src_audio_path = os.path.join(folder_paths.get_input_directory(), driving_audio)
        # if not os.path.exists(src_audio_path):
        #     src_audio_path = driving_audio # absolute path
        
        # save audio to path
        waveform = driving_audio["waveform"]
        sample_rate = driving_audio["sample_rate"]
        
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        
        src_audio_path = os.path.join(output_dir, f"hallo_{output_name}_src_audio.wav")
        torchaudio.save(src_audio_path, waveform, sample_rate)

        # === START OF MODIFIED SUBPROCESS EXECUTION ===
        python_executable = sys.executable
        
        # Get the site-packages directory for the current Python interpreter
        # and escape backslashes for embedding in Python code strings
        site_packages_path_escaped = ""
        for p in sys.path:
            if "site-packages" in p and os.path.isdir(p): # Make sure it's a directory
                site_packages_path_escaped = p.replace("\\", "\\\\")
                break
        
        # Escape cur_dir (path to ComfyUI_Hallo node) for embedding in Python code strings
        hallo_custom_node_dir_escaped = cur_dir.replace("\\", "\\\\")

        # --- Debug: Test import moviepy using a direct python -c call with sys.path manipulation ---
        import_test_code = f"import sys; "
        if site_packages_path_escaped: # Check if found
            import_test_code += f"sys.path.insert(0, r'{site_packages_path_escaped}'); "
        import_test_code += f"sys.path.insert(0, r'{hallo_custom_node_dir_escaped}'); " # For 'hallo' package itself
        import_test_code += "print(f'[Import Test] Subprocess sys.path: {{sys.path}}'); import moviepy; from moviepy.editor import AudioFileClip; print('moviepy.editor successfully imported in python -c test')"

        print(f"Attempting direct import test with code: {import_test_code}")
        try:
            test_import_cmd = [python_executable, "-c", import_test_code]
            # For this test, we won't try to set PYTHONPATH via env, 
            # as we are directly manipulating sys.path in the -c command.
            # Let it inherit the parent environment.
            test_result = subprocess.run(test_import_cmd, capture_output=True, text=True, check=False, cwd=cur_dir)
            print(f"[Import Test with Subprocess ENV] STDOUT: {test_result.stdout.strip()}")
            print(f"[Import Test with Subprocess ENV] STDERR: {test_result.stderr.strip()}")
        except Exception as e_test:
            print(f"[Import Test with Subprocess ENV] Failed: {e_test}")
        # === End of new debug ===

        # --- Prepare for the main inference.py call using runpy and sys.path manipulation ---
        # Path to inference.py, escaped for embedding in the Python code string
        inference_script_full_path_escaped = os.path.join(cur_dir, "scripts", "inference.py").replace("\\", "\\\\")

        # Arguments for inference.py (sys.argv for the script)
        script_args_for_runpy = [
            inference_script_full_path_escaped, # sys.argv[0] for inference.py
            '--config', tmp_yaml_path,
            '--source_image', src_img_path,
            '--driving_audio', src_audio_path, # Using src_audio_path from the context
            '--output', output_video_path,
            '--pose_weight', str(pose_weight),
            '--face_weight', str(face_weight),
            '--lip_weight', str(lip_weight),
            '--face_expand_ratio', str(face_expand_ratio)
        ]
        
        # Python code to execute via -c. This code will set up sys.path and then run inference.py
        runpy_code = f"""
import sys
import os
import runpy

# Escaped paths from the parent ComfyUI process
hallo_node_dir_for_subprocess = r'{hallo_custom_node_dir_escaped}'
site_packages_dir_for_subprocess = r'{site_packages_path_escaped}'

# Prepend paths for module searching
if site_packages_dir_for_subprocess and site_packages_dir_for_subprocess not in sys.path:
    sys.path.insert(0, site_packages_dir_for_subprocess)
if hallo_node_dir_for_subprocess and hallo_node_dir_for_subprocess not in sys.path:
    sys.path.insert(0, hallo_node_dir_for_subprocess)

print(f'[ComfyUI_Hallo nodes.py] Subprocess sys.path before runpy: {{sys.path}}')
print(f'[ComfyUI_Hallo nodes.py] Subprocess CWD: {{os.getcwd()}}')
print(f'[ComfyUI_Hallo nodes.py] Subprocess PYTHONPATH (from env): {{os.environ.get("PYTHONPATH", "Not Set")}} # AAAA marker')

sys.argv = {script_args_for_runpy!r} # Pass arguments to inference.py; !r gives a robust list representation
runpy.run_path(r'{inference_script_full_path_escaped}', run_name='__main__')
"""
        
        cmd_args_main = [python_executable, "-c", runpy_code]
        
        process_env = os.environ.copy() # Start with a clean copy of current environment
        # We are trying to control sys.path *inside* the subprocess via the -c code,
        # so setting PYTHONPATH here is currently omitted as per the strategy.

        print(f"Executing command: {python_executable} -c \"...python code...\" (see next line for details)")
        print(f"ComfyUI_Hallo: Python code for -c (main call): {runpy_code.replace(os.linesep, ' ').strip()}")
        print(f"ComfyUI_Hallo: Subprocess CWD for main call: {cur_dir}") # cur_dir is the unescaped path

        result = None # Initialize result
        try:
            # Execute the command for inference.py
            result = subprocess.run(cmd_args_main, env=process_env, capture_output=True, text=True, check=False, cwd=cur_dir) # Set CWD to custom node dir
            
            if result.returncode != 0:
                print("!!! Error during Hallo inference.py execution !!!")
                print(f"Return Code: {result.returncode}")
                print(f"Stdout:\n{result.stdout}") # Print stdout for more info
                print(f"Stderr:\n{result.stderr}")
                # raise RuntimeError(f"Hallo inference.py failed with stderr: {result.stderr}")
            else:
                print("Hallo inference.py executed successfully.")
                # print(f"Stdout: {result.stdout}")

        except FileNotFoundError:
            # For the error message, use the unescaped path to inference.py
            un_escaped_infer_py_path = os.path.join(cur_dir, "scripts", "inference.py")
            print(f"Error: Could not find Python executable at '{python_executable}' or script at '{un_escaped_infer_py_path}'")
            raise # Re-raise the exception to make it visible in ComfyUI
        except Exception as e:
            print(f"An unexpected error occurred while running inference.py: {e}")
            if result: # If subprocess was run and failed in an unexpected way but we have a result
                print(f"Subprocess STDOUT:\n{result.stdout}")
                print(f"Subprocess STDERR:\n{result.stderr}")
            raise # Re-raise

        # === END OF MODIFIED SUBPROCESS EXECUTION ===

        if not os.path.exists(output_video_path) or os.path.getsize(output_video_path) == 0:
             # This check is important. If the video wasn't created or is empty, cv_frame_generator will fail.
             # The error messages from subprocess.run (stderr) should give a clue why.
             if result and result.returncode !=0 : # if subprocess failed
                 raise RuntimeError(f"Hallo inference.py failed to produce a video. Check console for errors from inference.py. Stderr: {result.stderr}")
             else: # if subprocess seemed to succeed but video is still missing (edge case)
                 raise RuntimeError(f"Hallo inference.py did not produce the video file at {output_video_path}, or the file is empty. Subprocess stdout: {result.stdout if result else 'N/A'}")

        os.remove(tmp_yaml_path) # This should be fine here

        gen = cv_frame_generator(output_video_path)
        (width, height, target_frame_time) = next(gen)
        width = int(width)
        height = int(height)
        images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3)))))
        if len(images) == 0:
            raise RuntimeError("No frames generated")
        return (images, len(images), 25)

NODE_CLASS_MAPPINGS = {
    "D_HalloNode": HalloNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "D_HalloNode": "Hallo Node",
}