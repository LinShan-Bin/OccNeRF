import os
import cv2
import argparse

from tools.export_occupancy_vis import visualize_occ_dict


def create_video_from_images(input_folder, sem_only=False):
    output_video = input_folder + '.avi'
    image_files = [f for f in os.listdir(input_folder) if f.endswith(".jpg")]
    image_files.sort()

    frame = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'I420')
    if sem_only:
        out = cv2.VideoWriter(output_video, fourcc, 12.0, (width, height * 4))
    else:
        out = cv2.VideoWriter(output_video, fourcc, 12.0, (width, height * 3))

    num_frame = len(image_files) // 8
    for i in range(num_frame):
        image_up = cv2.imread(os.path.join(input_folder, os.path.join(input_folder, '{:03d}-up.jpg'.format(i))))
        image_down = cv2.imread(os.path.join(input_folder, os.path.join(input_folder, '{:03d}-down.jpg'.format(i))))

        occ_f = os.path.join(input_folder, '{:03d}-out.npy-front.jpg'.format(i))
        occ_fl = os.path.join(input_folder, '{:03d}-out.npy-front_left.jpg'.format(i))
        occ_fr = os.path.join(input_folder, '{:03d}-out.npy-front_right.jpg'.format(i))
        occ_up = cv2.hconcat([cv2.imread(occ_fl), cv2.imread(occ_f), cv2.imread(occ_fr)])
        occ_b = os.path.join(input_folder, '{:03d}-out.npy-back.jpg'.format(i))
        occ_bl = os.path.join(input_folder, '{:03d}-out.npy-back_left.jpg'.format(i))
        occ_br = os.path.join(input_folder, '{:03d}-out.npy-back_right.jpg'.format(i))
        occ_down = cv2.hconcat([cv2.imread(occ_bl), cv2.imread(occ_b), cv2.imread(occ_br)])
        
        frame = cv2.vconcat([image_up, occ_up, image_down, occ_down])
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a video from jpg images in a folder")
    parser.add_argument("input_folder", help="Input folder containing jpg images")
    parser.add_argument("--sem_only", action="store_true")
    args = parser.parse_args()
    
    dict_list = os.listdir(args.input_folder)
    dict_list.sort()
    for dict_name in dict_list:
        if dict_name.endswith('.npy'):
            visualize_occ_dict(os.path.join(args.input_folder, dict_name), offscreen=True, render_w=320)

    create_video_from_images(args.input_folder, sem_only=args.sem_only)
