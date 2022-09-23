import numpy as np

def first_fit_algorithm(weight_load, capacity):
    #荷物の数を得る
    n = len(weight_load)

    #indexの初期化
    i = 1 #箱の番号
    j = 1 #荷物の番号


    #箱の重さの配列の初期化
    weight_box_sum = [0]*n

    #荷物jを詰め込むことができる最小の添字の箱に詰め込む
    while j <= n:
        i = 1
        while True: #入る箱を探す
            if weight_box_sum[i-1] + weight_load[j-1] <= capacity:
                weight_box_sum[i-1] = weight_box_sum[i-1] + weight_load[j-1]
                j += 1
                break
            else:
                i += 1

    #中身のある箱の数を数える
    # box_num = sum([i > 0 for i in weight_box_sum])
    box_num = np.count_nonzero(weight_box_sum)

    return box_num #箱の個数

def main():
    #荷物の重量を配列として受け取る
    weight_list = list(map(int, input("荷物の重量をスペース区切りで複数入力して下さい ").split()))
    #荷物のキャパを入力して指定
    capacity = int(input("箱の重量制限を入力して下さい "))
    #重さのバリデーション
    if max(weight_list) > capacity:
        print("Ohh... someone is overweight...")
    else:
        print(first_fit_algorithm(weight_list, capacity))


if __name__ == "__main__":
    main()