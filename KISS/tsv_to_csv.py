import argparse, pathlib
import pandas as pd


def main(path):
  f_path = pathlib.Path(path)
  df = pd.read_csv(path, sep='\t')
  df1 = df.sort_values(['time','ue_id', 'serving_cell_id'], ascending=[True, True, True])
  print(df1.head(n=10))
  print(f_path)
  excel_path = f_path.with_suffix('.csv')
  print(excel_path)
  df1.to_csv(excel_path, index=False)

if __name__ == '__main__':  # run the main script

    # Create cmd line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=str, default=None, help='path of TSV file')
    
    # Create the args namespace
    args = parser.parse_args()

    # Run the __main__
    main(path=args.path)
