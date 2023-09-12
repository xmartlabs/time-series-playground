
from argparse import ArgumentParser
from clearml import Dataset


def main(args):

    # Create a dataset with ClearML`s Dataset class
    print(f"Creating dataset {args.dataset_name} in project {args.project}")
    dataset = Dataset.create(dataset_project=args.project, dataset_name=args.dataset_name)

    for f in args.files:
        # This works for both files and folders
        dataset.add_files(path=f)

    # Upload dataset to ClearML server (customizable)
    dataset.upload()

    # commit dataset changes
    dataset.finalize()


if __name__ == '__main__':
    # Example: python3 upload_dataset.py -p asd -n fds -f $BASE_PATH/*.csv $BASE_PATH/*/*.parquet
    parser = ArgumentParser()
    parser.add_argument('--project', '-p', type=str, default='Time Series PG')
    parser.add_argument('--dataset_name', '-n', type=str)
    parser.add_argument('--files', '-f', type=str, required=True, nargs='+')
    args = parser.parse_args()
    main(args)
