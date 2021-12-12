from pathlib import Path
import pandas

INPUT_DATASET = './dataset/train_set'             # Path to a folder that contains all training images
PATH_LABELS = './dataset/train_labels.csv'        # Path to CSV file containing training labels
PATH_DESTINATION = './dataset/train_set_labelled' # Where new folder will be created with dataset organised in folders by label

# Get a dict mapping file name to its label
# Example: {'train_1.jpg': 21}
file_label_dict = pandas.read_csv(PATH_LABELS).set_index('img_name').to_dict()['label']

for file in Path(INPUT_DATASET).iterdir():
    label = file_label_dict[file.name]
    dest_path = Path(PATH_DESTINATION) / str(label)
    # Create missing directories if necessary
    dest_path.mkdir(parents=True, exist_ok=True)
    # Copy the file to the folder with a correct label
    file.rename(dest_path / file.name)
