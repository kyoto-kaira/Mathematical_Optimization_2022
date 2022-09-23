#荷物の重さが入った配列を受け取る

def next_fit_algorithm(weight_load, capacity):
    #荷物の数を得る
    n = len(weight_load)

    #indexの初期化
    i = 1 #箱の番号
    j = 1 #荷物の番号

    #箱の重さの配列の初期化
    weight_box_sum = [0]*n

    #各箱の容量を超えないようにぶち込んでいく
    while j <= n:
        if weight_box_sum[i-1] + weight_load[j-1] <= capacity:
            weight_box_sum[i-1] = weight_box_sum[i-1] + weight_load[j-1]
            j += 1
        else:
            i += 1

    return i #箱の個数

def main():
    #荷物の重量を配列として受け取る
    weight_list = list(map(int, input("荷物の重量をスペース区切りで複数入力して下さい ").split()))
    #荷物のキャパを入力して指定
    capacity = int(input("箱の重量制限を入力して下さい "))
    print(next_fit_algorithm(weight_list, capacity))


if __name__ == "__main__":
    main()