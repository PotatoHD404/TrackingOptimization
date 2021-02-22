filename = r'C:\Users\korna\Downloads\data1.txt'

if __name__ == '__main__':
    file1 = open(filename, 'r')
    content = file1.readlines()
    for s in content:
        content[content.index(s)] = list(map(float, s.replace('\n', '').split(' ')))
    content.sort()
    res = []
    for i in range(1, 1001):
        res.append([0.0, 0.0, 0])
        for j in content:
            if -0.5 <= j[0] - i < 0.5:
                res[-1][0] += j[0]
                res[-1][1] += j[1]
                res[-1][2] += 1
        if res[-1][2] != 0:
            res[-1][0] /= res[-1][2]
            res[-1][1] /= res[-1][2]

    for i in res:
        if i[0] != 0.0:
            print(f'{i[0]} {i[1]}')
