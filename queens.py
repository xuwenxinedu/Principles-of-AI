import numpy as np
import random
import time

def ruleset(seqs):
    """
    查看某个状态节点seqs（棋盘上的一行）皇后布局是否满足规则，即列和对角线是否有其他皇后    
    """
    a = np.array([0] * 81)  
    a = a.reshape(9, 9)  # 初始化空白棋盘
    n = 0  

    for i in range(1, 9):
        if seqs[i-1] != 0: # seqs的某一元素为0代表对应棋盘的该列不应该放置任何皇后
            a[seqs[i - 1]][i] = 1  # 根据序列，按从第一列到最后一列的顺序，在空白棋盘对应位置放一个皇后，生成当前序列对应的棋盘

    for i in range(1, 9):
        if seqs[i - 1] == 0:
            continue 
        for k in list(range(1, i)) + list(range(i + 1, 9)):  # 检查每个皇后各自所在的行上是否有其他皇后
            if a[seqs[i - 1]][k] == 1:  # 有其他皇后
                n += 1
        t1 = t2 = seqs[i - 1]
        for j in range(i - 1, 0, -1):  # 看左半段的两条对角线
            if t1 != 1:
                t1 -= 1
                if a[t1][j] == 1:
                    n += 1  

            if t2 != 8:
                t2 += 1
                if a[t2][j] == 1:
                    n += 1  

        t1 = t2 = seqs[i - 1]
        for j in range(i + 1, 9):  # 看右半段的两条对角线
            if t1 != 1:
                t1 -= 1
                if a[t1][j] == 1:
                    n += 1 

            if t2 != 8:
                t2 += 1
                if a[t2][j] == 1:
                    n += 1  
    return int(n/2)  # 返回n/2，因为A攻击B也意味着B攻击A，因此返回n的一半

def display_board(seqs):
    """
     显示解序列seqs对应的棋盘
    """
    board = np.array([0] * 81)  
    board = board.reshape(9, 9)  
    
    for i in range(1, 9):
        board[seqs[i - 1]][i] = 1  
    print('对应棋盘如下:')
    for i in board[1:]:
        for j in i[1:]:
            print(j, ' ', end="")  
        print()  
    print('攻击的皇后对数为' + str(ruleset(seqs)))






def DFS():
    '''
    执行深度优先搜索算法
    '''
    print("0")
    start = time.time()
    openlist = [[0] * 8] # 使用栈作为open表
    solution = []
    flag = 0 # 代表还未找到解
    print("1")

    while openlist: 
        if flag == 1: # 找到解就退出循环
            break
        seqs = openlist.pop(-1) # LIFO，先扩展最新加入栈的序列（节点）
        nums = list(range(1, 9))  
        for j in range(8): 
            pos = seqs.index(0)
            temp_seqs = list(seqs)
            temp = random.choice(nums)                     # 在该列随机挑选一行放置皇后
            temp_seqs[pos] = temp                           # 将皇后放在该列的第temp行
            del nums[nums.index(temp)]                           # 从nums移除已产生的值
            if ruleset(temp_seqs) == 0:  
                openlist.append(temp_seqs)                       # 将皇后放在该列的第temp行后，若序列对应棋盘无互相攻击的皇后，则将序列存储到openlist
                if 0 not in temp_seqs: 
                    solution = temp_seqs                   # 生成节点时做goal test：若序列中无0元素，即八个皇后均已放好，则序列为解序列
                    flag = 1 # 成功
                    break

    if solution:
        print('已找到解序列：' + str(solution))
        display_board(solution)
    else:
        print('算法失败，未找到解')

    end = time.time()
    print('用时' + str('%.2f' % (end-start)) + 's')
    
if __name__ == '__main__':
    DFS()
