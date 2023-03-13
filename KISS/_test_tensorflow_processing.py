import tensorflow as tf
import multiprocessing

def process_data(data):
    # your data processing logic here
    print(data)
    pass

if __name__ == '__main__':
    data = list(range(123))  # your data here
    max_workers = 4  # set the maximum number of processes to use
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # if there are GPUs available, set memory growth and use the first GPU
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.set_visible_devices(gpus[0], 'GPU')
        except RuntimeError as e:
            print(e)
    with multiprocessing.Pool(max_workers) as pool:
        # map each task to a separate process
        results = pool.map_async(process_data, data)
        # wait for all processes to finish and collect the results
        results.wait()
