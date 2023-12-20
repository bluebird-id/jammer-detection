import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Conv1D, Conv2D, Flatten, Reshape, MaxPooling1D, MaxPooling2D, BatchNormalization, TimeDistributed
from tensorflow.keras.optimizers.legacy import Adam
import time, math
import numpy as np
from functools import lru_cache

def find_lr(model, train_dataset, loss_func):

    # sample batches once
    sample_batches = train_dataset.take(3)

    # snapshot initial weights to be restored each check
    initial_weights = model.get_weights()

    @lru_cache
    def loss_at(log_learning_rate):
        # reset model
        model.set_weights(initial_weights)
        learning_rate = 10 ** log_learning_rate
        opt = Adam(learning_rate=learning_rate)
        model.compile(loss=loss_func, optimizer=opt)
        # run a small number of batches
        t_start = time.perf_counter()
        losses = []
        for x, y in sample_batches:
            loss = model.train_on_batch(x, y)
            losses.append(loss)
        # print("t_train_sample", time.perf_counter()-t_start)
        return np.mean(losses)

    def golden_section_search(f, a, b, max_iters=10, max_time=60):
        # https://en.wikipedia.org/wiki/Golden-section_search
        n_iter = 0
        t_start = time.perf_counter()
        gr = (math.sqrt(5) + 1) / 2
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        done = False
        while not done:
            n_iter += 1
            t_running = time.perf_counter() - t_start
            # print("a", a, "b", b,
            #       "n_iter", n_iter, "t_running", t_running)
            print('.', end='')
            if f(c) < f(d):
                b = d
            else:
                a = c
            c = b - (b - a) / gr
            d = a + (b - a) / gr
            done = (n_iter > max_iters) or (t_running > max_time)
        return (b + a) / 2


    print('Finding optimal learning rate', end='')
    t_start = time.perf_counter()
    final_log_lr = golden_section_search(loss_at, -5, -1)
    final_lr = 10 ** final_log_lr
    print("\nfinal_log_lr", final_log_lr,
          "=> final_lr", final_lr,
          "in ", time.perf_counter()-t_start, "sec")

    return final_lr
