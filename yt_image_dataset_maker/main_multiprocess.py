import multiprocessing as mp

from main import read_dict_from_csv, full_pipe_line


def multiprocess_full_pipe_line(csv_file_path: str) -> None:
    cores = int(mp.cpu_count() - 4)
    data_dicts = read_dict_from_csv(csv_file_path)
    with mp.Pool(processes=cores) as pool:
        pool.map(full_pipe_line, data_dicts)


if __name__ == "__main__":
    csv_file_path = "data.csv"
    multiprocess_full_pipe_line(csv_file_path)
