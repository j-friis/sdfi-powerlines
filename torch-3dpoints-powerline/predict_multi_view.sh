#!/bin/bash
# Default values for the arguments
n_process=3
data_path="~/data"
height_filter=3.5
model_path="~/Documents/powerline/torch-3dpoints-powerline/outputs/2023-06-12/23-43-09/SEUNet18.pt"
model_metric="miou"
cnn2d_path="~/Documents/powerline/torch-3dpoints-powerline/models/preprocess_CNN/cnnStateDict.pth"

# Function to display the help information
display_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --n_process     <value>      The number of process to do PDAL pipeline (default: $n_process)"
    echo "  --data          <value>      The path to the folder containing the denmark folder (default: $data_path)"
    echo "  --height_filter  <value>      The height in meters to remove from the bottom (default: $height_filter)"
    echo "  --model_path    <value>      The path to the model (default: $model_path)"
    echo "  --model_metric  <value>      What metric to choose the model from it options are latest, loss_seg, acc, macc and miou (default: $model_weight)"
    echo "  --cnn2d_path    <value>      The path to the 2D cnn that operates on the rasterized images (default: $cnn2d_path)"
    echo "  --help                       Display this help message"
    # Add more options and their descriptions as needed
}

# Process the command-line options and arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --n_process)
            n_process="$2"
            shift 2
            ;;
        --data)
            data="$2"
            shift 2
            ;;
        --height_filter)
            height_filter="$2"
            shift 2
            ;;
        --model_path)
            model_path="$2"
            shift 2
            ;;
        --model_metric)
            model_metric="$2"
            shift 2
            ;;
        --cnn2d_path)
            cnn2d_path="$2"
            shift 2
            ;;
        --help)
            display_help  # Call the display_help function
            exit 0        # Exit the script successfully
            ;;
        --h)
            display_help  # Call the display_help function
            exit 0        # Exit the script successfully
            ;;
        -help)
            display_help  # Call the display_help function
            exit 0        # Exit the script successfully
            ;;
        -h)
            display_help  # Call the display_help function
            exit 0        # Exit the script successfully
            ;;
        *)
            echo "Invalid option: $1" >&2
            exit 1
            ;;
    esac
done

# Shift the processed options
shift $((OPTIND - 1))

echo "Runing pdal pipeline"
eval "$(conda shell.bash hook)"

path_to_test_data="$data_path/denmark/raw/test"
echo "$path_to_test_data"
echo "Runing pdal pipeline"
conda activate pdal
# python torch_points3d/core/data_transform/pdal/run_pipeline.py $path_to_test_data $n_process $height_filter 0.1

echo "Predecting with 3D CNN"
conda activate msc
python eval_to_las.py $data_path $model_path $model_metric $cnn2d_path
