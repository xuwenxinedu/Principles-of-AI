from kanren import *
from kanren.core import lall
#定义left函数用来查找房屋左边
def left(q,p,list):
	return membero((q,p),zip(list,list[1:]))
#定义next接近谁的房子
def next(q,p,list):
	return conde([left(q,p,list)],[left(p,q,list)])
#声明房子变量
houses = var()

#给出条件句
rules_zebraproblem = lall(
    (eq, (var(), var(), var(), var(), var()), houses),      # 5个var()分别代表人、烟、饮料、动物、屋子颜色
    # 房子里的每个子成员有五个属性: membero(国家，身份，饮料，宠物，房子)
    (membero,('英国人', var(), var(), var(), '红色'), houses),         # 1. 英国人住在红色的房子里
    (membero,('西班牙人', var(), var(), '狗', var()), houses),         # 2. 西班牙人养了一条狗
    (membero,('日本人', '油漆工', var(), var(), var()), houses),       # 3. 日本人是一个油漆工
	(membero,('意大利人', var(), '茶', var(), var()), houses),         # 4. 意大利人喜欢喝茶
    (eq,(('挪威人', var(), var(), var(), var()), var(), var(), var(), var()), houses),# 5. 挪威人住在左边的第一个房子里 
    (left,(var(), var(), var(), var(), '白色'),(var(), var(), var(), var(), '绿色'), houses), # 6. 绿房子在白房子的右边
    (membero,(var(), '摄影师', var(), '蜗牛', var()), houses),                    # 7. 摄影师养了一只蜗牛
    (membero,(var(), '外交官', var(), var(), '黄色'), houses),                    # 8. 外交官住在黄房子里
    (eq,(var(), var(), (var(), var(), '牛奶', var(), var()), var(), var()), houses),# 9. 中间那个房子的人喜欢喝牛奶
	(membero,(var(), var(), '咖啡', var(), '绿色'), houses),                       # 10. 喜欢喝咖啡的人住在绿房子里
    (next,('挪威人', var(), var(), var(), var()),(var(), var(), var(), var(), '蓝色'), houses),# 11. 挪威人住在蓝色的房子旁边
	(membero,(var(), '小提琴家', '橘子汁', var(), var()), houses),              # 12. 小提琴家喜欢喝橘子汁
	(next,(var(), var(), var(), '狐狸', var()),(var(), '医生', var(), var(), var()), houses), # 13. 养狐狸的人所住的房子与医生的房子相邻
    (next,(var(), var(), var(), '马', var()),(var(), '外交官', var(), var(), var()), houses),  # 14. 养马的人所住的房子与外交官的房子相邻
    (membero,(var(), var(), var(), '斑马', var()), houses),                 # 有人养斑马
    (membero,(var(), var(), '矿泉水', var(), var()), houses),               # 有人喜欢喝矿泉水
)
# 使用rules_zebraproblem约束运行解算器
solutions = run(0, houses, rules_zebraproblem)
# 提取解算器的输出
output = [house for house in solutions[0] if '斑马' in house][0][4]
print ('\n{}房子里的人养斑马'.format(output))
output = [house for house in solutions[0] if '矿泉水' in house][0][4]
print ('\n{}房子里的人喜欢喝矿泉水'.format(output))
