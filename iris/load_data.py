import pandas as pd
import argparse

if __name__ == "__main__":
    
    argument_parser = argparse.ArgumentParser()
    
    argument_parser.add_argument(
        '--data_path', type=str,
        default='./Iris.csv',
        help="Input data path"
    )
    
    args = argument_parser.parse_args()
    data = pd.read_csv(args.data_path)
    print(data.shape)
    
    print("load_data")
    
    data.to_csv('./Iris.csv', index=False)