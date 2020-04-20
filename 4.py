import math
import sys
class node:
    purity = 0
    label = ""
    left_node = None
    right_node = None
    data = []
    children = []
    split_point = 0
    split_index = 0

    def __init__(self, label="", data=[], purity=0):
        self.label = label
        self.data = data
        self.purity = purity

#可能分支
def P_ciD_Y(n_vi, n_v):
    sum = 0
    for j in range(len(n_v)):
        # #所有点的和
        sum += n_v[j]
        # #Y中点除以所有点
    ans = n_vi/sum
    return ans

# #N中可能性
def P_ciD_N(n_vi, n_v, n_ii, n_i):
    sum = 0
    for j in range(len(n_v)):
        # #所有的点减去Y点
        sum += n_i[j] - n_v[j]
    if sum == 0:
        return 0
    return (n_ii-n_vi)/sum

# #D中可能性
def P_ciD(c_index, D,C):
    c_sum = 0
    for i in range(len(D)):
        if D[i]["label"] == C[c_index]:  # #由标签知道是哪一类
            c_sum += 1

    return c_sum/len(D)

# #D中ci的熵函数
def Entropy(p):
    Entropy_sum = 0
    for i in range(len(p)):
        if p[i] == 0.0:
            continue
        Entropy_sum -= p[i]*math.log(p[i], 2)

    return Entropy_sum

# #信息增益函数
def Gain(G_D,G_DY,G_DN,pd,py,pn):
    n = len(G_D)
    n_Y = len(G_DY)
    n_N = len(G_DN)
    gain = Entropy(pd) - n_Y/n*Entropy(py)-n_N/n*Entropy(pn)
    return gain
#纯净度
def purity(n_i,n):
    temp = 0
    temp_index = 0
    for i in range (len(n_i)):
        if n_i[i]/n > temp:
            temp = n_i[i]/n
            temp_index = i
    return temp, temp_index

def Evaluate_NU_umeric_Attribute(data_lis, x_i,class_cat):
	n_v = {}
	data_lis = sorted(data_lis, key=lambda x: x["data"][x_i])
	M = set()
	n_i = [0]*len(class_cat)    # #初始化类个数
	for j in range(len(data_lis)-1):  # #D中除最后一个的每个点
		for i in range(len(class_cat)):  # #判断属于哪一类
			if data_lis[j]["label"] == class_cat[i]:
				n_i[i] += 1  # #找到该类 点数加1
				#print("work")
		if data_lis[j+1]["data"][x_i] != data_lis[j]["data"][x_i]:
			v = (data_lis[j+1]["data"][x_i] + data_lis[j]["data"][x_i])/2  # #中间点
			M.add(v)  # #加入集合
			n_v2 = []
			for k in range(len(class_cat)):
				n_v2.append(n_i[k])
			n_v[v]=(n_v2)
			#print("n_i[i]",n_i)
			#print("n_v",n_v)
			#print("n_v2",n_v2 )
	#print(n_v)
	#sys.exit()
	

	for i in range(len(class_cat)):
		if data_lis[len(data_lis)-1]["label"] == class_cat[i]:
			n_i[i] += 1
			break   

	v_star = 0
	score_star = 0
	#print("n_v",n_v)
	#print("n_i",n_i)
	
	'''
	p_ciD_Y=[0]*len(class_cat)
	p_ciD_N=[0]*len(class_cat)
	'''
	for v in M:  # #对集合中的每个midpoint进行判断
		p_ciD_Y=[]
		p_ciD_N=[]
		p_ciD_D=[]
		for i in range(len(class_cat)):
			'''
			p_ciD_Y = P_ciD_Y(n_v[i], n_v)  # #属于ci类且为Y的可能性
			p_ciD_N = P_ciD_N(n_v[i], n_v, n_i[i], n_i)  # #属于ci类且为N的可能性
			'''
			p_ciD_D.append(P_ciD(i,data_lis,class_cat))
			p_ciD_Y.append(P_ciD_Y(n_v[v][i], n_v[v]))
			p_ciD_N.append(P_ciD_N(n_v[v][i], n_v[v], n_i[i], n_i))
		#print("p_ciD_D",p_ciD_D)
		#print("p_ciD_Y",p_ciD_Y)
		#print("p_ciD_N",p_ciD_N)
		#if p_ciD_D[0] < 0.33 or p_ciD_D[0] >0.34:
		#sys.exit()
		DY = []  # #记录Y中的点
		DN = []  # #记录N中的点
		for j in range(len(data_lis)):
			if data_lis[j]["data"][x_i] <= v:  # #小于V即为Y
				DY.append(data_lis[j])  # #插入列表中
			else:
				DN.append(data_lis[j])

		score_minv = Gain(data_lis, DY, DN,p_ciD_D,p_ciD_Y,p_ciD_N)  # #算出得分
		#print("v",v)
		#print("score_minv",score_minv)
		if score_minv > score_star:
			v_star = v
			score_star = score_minv
	return v_star,score_star

def DecisionTree(data_lis, leaf_size, is_pure, root,class_cat):
	n = len(data_lis)
	n_i = [0]*len(class_cat)
	for i in range(len(class_cat)):
		for j in range(n):
			if data_lis[j]["label"] == class_cat[i]:
				n_i[i] += 1
	purity_D, purity_index = purity(n_i, n)  # #计算纯净度
	print("\n---------------------------------------------------------------")
	print("--> a new tree start ...")
	print("purity to now tree:", purity_D,end=" ")
	print("(",purity_index,")")
	print("size to now tree:",len(data_lis))
	if n <= leaf_size or purity_D > is_pure:  # #如果小于叶节点数或纯净度大于要求就转化为叶节点
		print("\n************************************************")
		print("*TIME TO STOP :is_pure or small than leaf_size *")
		print("*the index that fullfill the standard:",purity_index,"      *")
		print("************************************************")
		C_star = class_cat[purity_index]
		root.C_label = C_star
		root.data = data_lis
		root.purity = purity_D
		return
	print("--> begin to calc ...")
	splitpoint_star = 0
	score_star = 0
	split_index = 0
	for i in range(len(data_lis[0]["data"])-1):  # #对每个数字属性进行评估打分
		v, score = Evaluate_NU_umeric_Attribute(data_lis,i,class_cat)

		#让score为最大得分
		#print("score",i,v,score)
		if score > score_star:
			score_star = score
			splitpoint_star = v
			split_index = i
	#sys.exit()
	root.splitpoint_star = splitpoint_star
	root.splitpoint_index = split_index
	#输出internal node
	print("internal node star :", splitpoint_star)
	print("internal node index :", split_index)
	Dy = []
	Dn = []
	for i in range(len(data_lis)):  # #将Y和N中的点存入列表中

		if data_lis[i]["data"][split_index] <= splitpoint_star:
			Dy.append(data_lis[i])
		else:
			Dn.append(data_lis[i])
	if len(Dy) >1:
		root.left = node()
		#print("left-len:",len(Dy))
		DecisionTree(Dy, leaf_size, is_pure, root.left,class_cat)
		
	if len(Dn) >1:
		root.right = node()
		#print("right-len:",len(Dn))
		DecisionTree(Dn, leaf_size, is_pure, root.right,class_cat)
		
	#print("test! Dy", len(Dy), Dy)
	#print("test1 Dn", len(Dn), Dn)

def readFile(addr):
	f = open(addr, "r", encoding="utf-8")
	data_lis = []
	class_lis = []

	for line in f:
		data_dic = {}
		data_dic['data']=[]
		s = line.strip().split(',')
		if s[0] is '':continue
		label = s[-1].strip('"')
		if label not in class_lis:
			class_lis.append(label)
		for i in range(len(s)-1):
			data_dic['data'].append(float(s[i]))
		data_dic['label'] = label
		data_lis.append(data_dic)
	return data_lis,class_lis
			
if __name__ == "__main__":
	data,class_ =readFile("iris.txt")
	#print(data)
	#print(len(data[0]["data"]))
	root = node()
	DecisionTree(data, 5, 0.95, root,class_)